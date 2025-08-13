import requests
import json
import os
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv
import pinecone
from openai import OpenAI
import re
import tiktoken
from langchain_community.graphs import Neo4jGraph
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Explicitly load .env file from the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

def fetch_papers(topic, limit=100):
    """
    Fetches papers from the Semantic Scholar Graph API on a given topic.

    Args:
        topic (str): The topic to search for.
        limit (int): The maximum number of papers to fetch.

    Returns:
        list: A list of dictionaries, where each dictionary represents a paper.
    """
    print(f"Fetching {limit} papers on '{topic}'...")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={topic.replace(' ', '+')}&limit={limit}&fields=title,abstract,citationCount,references.paperId"
    
    papers_data = None
    retries = 5
    backoff_factor = 2 # Starting with a longer backoff
    for i in range(retries):
        try:
            print(f"Attempt {i+1} of {retries}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            papers_data = response.json()
            print("Successfully fetched data.")
            break
        except requests.exceptions.RequestException as e:
            print(f"An error occurred on attempt {i+1}: {e}")
            wait_time = backoff_factor * (2 ** i)
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Non-rate-limit error. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

    if not papers_data:
        print("Failed to fetch papers after several retries.")
        return []

    papers = []
    if papers_data.get('data'):
        for item in papers_data.get('data', []):
            title = item.get('title', 'No Title')
            abstract = item.get('abstract', 'No Abstract')
            references_list = item.get('references')
            references = []
            if references_list:
                for ref in references_list:
                    if ref and 'paperId' in ref:
                        references.append(ref['paperId'])
            paper = {
                'paperId': item['paperId'],
                'title': title,
                'abstract': abstract,
                'citationCount': item.get('citationCount'),
                'references': references,
                # Construct a stable source URL for provenance in the UI
                'url': f"https://www.semanticscholar.org/paper/{item['paperId']}"
            }
            papers.append(paper)
    
    if not os.path.exists('data'):
        os.makedirs('data')

    with open('data/papers.json', 'w') as f:
        json.dump(papers, f, indent=2)
    
    print(f"Successfully saved {len(papers)} papers to data/papers.json")
    return papers

def ensure_constraints() -> None:
    """Create id/name/date constraints to keep the graph clean and fast."""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, password]):
        return
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paperId IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Grant) REQUIRE g.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:State) REQUIRE s.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Deadline) REQUIRE d.date IS UNIQUE")
    driver.close()


def load_to_neo4j(papers, dataset: str = None):
    """
    Loads papers and their citations into a Neo4j database.

    Args:
        papers (list): A list of paper dictionaries.
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        print("Neo4j credentials not found in .env file. Skipping Neo4j load.")
        return

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # Create papers (skip invalid)
        for paper in papers:
            paper_id = paper.get('paperId')
            if not paper_id:
                continue
            session.run("""
                MERGE (p:Paper {paperId: $paperId})
                SET p.title = $title, p.citationCount = $citationCount, p.url = $url,
                    p.dataset = coalesce($dataset, p.dataset)
            """, paperId=str(paper_id), title=paper.get('title'), citationCount=paper.get('citationCount'), url=paper.get('url'), dataset=dataset)

        # Create citation relationships (create stub target if needed to improve connectivity)
        for paper in papers:
            paper_id = paper.get('paperId')
            if not paper_id:
                continue
            if paper.get('references'):
                for ref_id in paper['references']:
                    if not ref_id:
                        continue
                    session.run("""
                        MATCH (p1:Paper {paperId: $p1_id})
                        MERGE (p2:Paper {paperId: $p2_id})
                        ON CREATE SET p2.stub = true, p2.dataset = coalesce($dataset, p2.dataset)
                        MERGE (p1)-[r:CITES]->(p2)
                        ON CREATE SET r.source = 'etl', r.dataset = $dataset
                    """, p1_id=str(paper_id), p2_id=str(ref_id), dataset=dataset)
        
        print(f"Successfully loaded {len(papers)} papers and their citations into Neo4j.")

    driver.close()


def add_topic_edges(papers, topic_name: str, dataset: str = None) -> None:
    """
    Deterministically attach each paper in this ingest batch to a Topic via (:Paper)-[:HAS_TOPIC]->(:Topic).

    This creates the minimum structure needed for topic questions like
    "most cited paper on X" to work reliably.
    """
    if not topic_name:
        return

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, password]):
        print("Neo4j credentials not found in .env file. Skipping HAS_TOPIC edges.")
        return

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("""
            MERGE (t:Topic {name: $name})
            ON CREATE SET t.dataset = coalesce($dataset, t.dataset)
        """, name=topic_name, dataset=dataset)

        for paper in papers:
            session.run("""
                MATCH (p:Paper {paperId: $paperId})
                MERGE (t:Topic {name: $name})
                MERGE (p)-[r:HAS_TOPIC]->(t)
                ON CREATE SET r.source = 'etl', r.dataset = $dataset
            """, paperId=paper['paperId'], name=topic_name, dataset=dataset)

    driver.close()
    print(f"Attached {len(papers)} papers to Topic '{topic_name}' via HAS_TOPIC.")

def load_to_pinecone(papers):
    """
    Computes embeddings for papers and upserts them into a Pinecone index.

    Args:
        papers (list): A list of paper dictionaries.
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([pinecone_api_key, openai_api_key]):
        print("Pinecone or OpenAI API key not found in .env file. Skipping Pinecone load.")
        return

    pc = pinecone.Pinecone(api_key=pinecone_api_key)
    openai_client = OpenAI(api_key=openai_api_key)

    index_name = "hybrid-research-poc"
    if index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name, 
            dimension=1536, 
            metric="cosine",
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1') # Free tier region
        )
    
    index = pc.Index(index_name)

    print("Upserting vectors to Pinecone...")

    def _truncate_for_embedding(text: str, max_tokens: int = 7500) -> str:
        """Trim long texts to fit within the embedding model context safely."""
        if not text:
            return ""
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            try:
                enc = tiktoken.encoding_for_model("text-embedding-3-small")
            except Exception:
                # Fallback: crude character cut
                return text[:24000]
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    for paper in papers:
        if paper.get('abstract'):
            text_to_embed = _truncate_for_embedding(f"{paper['title']}: {paper['abstract']}")
            
            try:
                print(f"  - Embedding and upserting '{paper['paperId']}'")
                res = openai_client.embeddings.create(input=[text_to_embed], model="text-embedding-3-small")
                embedding = res.data[0].embedding

                index.upsert(vectors=[{
                    "id": paper['paperId'],
                    "values": embedding,
                    "metadata": {"graph_id": f"Paper:{paper['paperId']}"}
                }])
            except Exception as e:
                print(f"    Failed to process paper {paper['paperId']}: {e}")

    print("Successfully loaded papers into Pinecone.")

def run_llm_citation_experiment(papers):
    """
    Uses an LLM to extract citation counts from paper abstracts and updates Neo4j.
    This is a direct test of the LLM's ability to extract specific, verifiable data.
    """
    print("Running LLM Citation Extraction Experiment...")
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([uri, user, password, openai_api_key]):
        print("Missing credentials for experiment. Aborting.")
        return

    driver = GraphDatabase.driver(uri, auth=(user, password))
    openai_client = OpenAI(api_key=openai_api_key)

    with driver.session() as session:
        for i, paper in enumerate(papers):
            if not paper.get('abstract'):
                continue

            # We add a simulated sentence to the abstract to give the LLM a chance.
            # In a real scenario, this might come from parsing a PDF's reference section.
            text_to_process = (
                f"Title: {paper['title']}. Abstract: {paper['abstract']}. "
                f"This paper has been cited {paper['citationCount']} times."
            )
            
            prompt = (
                "From the following text, extract the citation count as an integer. "
                "If no citation count is mentioned, you must return 0. "
                "Only return the integer number, with no other text or explanation.\n\n"
                f"Text: \"{text_to_process}\""
            )

            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                llm_result = response.choices[0].message.content
                # Clean up the result to ensure it's just a number
                citation_num_str = re.search(r'\d+', llm_result)
                if citation_num_str:
                    llm_citations = int(citation_num_str.group(0))
                else:
                    llm_citations = 0

                print(f"  ({i+1}/{len(papers)}) Paper {paper['paperId']}: Ground Truth={paper['citationCount']}, LLM says={llm_citations}")

                # Update the node in Neo4j with the LLM's finding
                session.run("""
                    MATCH (p:Paper {paperId: $paperId})
                    SET p.llm_extracted_citations = $llm_citations
                """, paperId=paper['paperId'], llm_citations=llm_citations)

            except Exception as e:
                print(f"    Failed to process paper {paper['paperId']} with LLM: {e}")
    
    driver.close()
    print("LLM Citation Extraction Experiment complete.")


def build_graph_with_langchain(papers, dataset: str = None):
    """
    Uses LangChain's LLMGraphTransformer to build a knowledge graph
    from paper abstracts and load it into Neo4j.
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([uri, user, password, openai_api_key]):
        print("Missing credentials for LangChain graph builder. Aborting.")
        return

    print("Building knowledge graph with LangChain...")

    # 1. Connect to Neo4j
    graph = Neo4jGraph(url=uri, username=user, password=password)

    # 2. Define the graph schema (ontology)
    # This gives the LLM guardrails for what to extract.
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Paper", "Author", "Topic", "Keyword"],
        # Enable HAS_TOPIC so the builder can attach topics inferred from text
        allowed_relationships=["CITES", "AUTHORED_BY", "HAS_TOPIC", "MENTIONS", "RELATED_TO"]
    )

    # 3. Process papers in batches
    documents = []
    for paper in papers:
        if paper.get('abstract'):
            doc = Document(
                page_content=f"Title: {paper['title']}. Abstract: {paper['abstract']}",
                metadata={'source': paper['paperId']}
            )
            documents.append(doc)

    print(f"Processing {len(documents)} documents with the LLMGraphTransformer...")
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # 4. Load the graph into Neo4j
    print("Loading extracted graph into Neo4j...")
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

    print("Successfully built and loaded graph using LangChain.")

    # 5. Annotate LLM-created structures with provenance and dataset, without overwriting ETL
    print("Annotating inferred nodes and relationships with provenance...")
    with graph._driver.session() as session:  # type: ignore[attr-defined]
        session.run(
            """
            // set dataset on nodes if missing
            MATCH (n)
            WHERE n.dataset IS NULL
            SET n.dataset = $dataset
            """,
            dataset=dataset,
        )
        session.run(
            """
            // mark LLM-created relationships where source is not already set
            MATCH ()-[r]->()
            WHERE r.source IS NULL
            SET r.source = 'llm', r.model = 'gpt-4o-mini', r.dataset = $dataset, r.confidence = coalesce(r.confidence, 0.7)
            """,
            dataset=dataset,
        )

        # Normalize and merge duplicate Topics by case-insensitive name
        session.run(
            """
            CALL apoc.periodic.iterate(
              'MATCH (t:Topic) WITH toLower(trim(t.name)) AS key, collect(t) AS nodes RETURN key, nodes',
              'WITH key, nodes WHERE size(nodes) > 1
               CALL apoc.refactor.mergeNodes(nodes, {properties:"combine"}) YIELD node
               SET node.name = key
               RETURN node', {batchSize:100, parallel:false})
            YIELD batches RETURN batches
            """
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Fetch papers from Semantic Scholar and load to Neo4j and Pinecone.")
    parser.add_argument("--topic", type=str, default="graph neural networks", help="Topic to search for papers on.")
    parser.add_argument("--limit", type=int, default=100, help="Number of papers to fetch.")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching and only load from existing json file.")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip loading to Neo4j (manual method).")
    parser.add_argument("--skip-pinecone", action="store_true", help="Skip loading to Pinecone.")
    parser.add_argument("--run-llm-experiment", action="store_true", help="Run the LLM citation extraction experiment.")
    parser.add_argument("--use-langchain-builder", action="store_true", help="Use the LangChain LLMGraphTransformer to build the graph.")
    parser.add_argument("--no-deterministic-topic-edges", action="store_true", help="Do not attach HAS_TOPIC edges using the CLI topic; rely solely on LLM-inferred topics.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset tag to attach to nodes/edges for multi-tenant separation.")
    args = parser.parse_args()

    if not args.skip_fetch:
        papers_list = fetch_papers(args.topic, args.limit)
    else:
        print("Skipping fetch. Loading papers from data/papers.json")
        if os.path.exists("data/papers.json"):
            with open("data/papers.json", "r") as f:
                papers_list = json.load(f)
        else:
            print("data/papers.json not found. Please run without --skip-fetch first.")
            papers_list = []

    if papers_list:
        # Compute default dataset name if not provided: e.g., papers_graph_neural_networks
        default_dataset = args.dataset or f"papers_{args.topic.replace(' ', '_').lower()}"
        # Ensure fundamental constraints exist before writes
        ensure_constraints()
        # Optionally run both: first load ground-truth relationships, then enrich with LLM
        if not args.skip_neo4j:
            load_to_neo4j(papers_list, dataset=default_dataset)
            # Optionally attach deterministic topic edges for the fetched topic
            # Controlled by CLI flag --no-deterministic-topic-edges
            try:
                deterministic_flag = getattr(args, 'no_deterministic_topic_edges', False)
            except Exception:
                deterministic_flag = False
            if not deterministic_flag:
                add_topic_edges(papers_list, args.topic, dataset=default_dataset)
        if args.use_langchain_builder:
            build_graph_with_langchain(papers_list, dataset=default_dataset)

        if not args.skip_pinecone:
            load_to_pinecone(papers_list)
        if args.run_llm_experiment:
            run_llm_citation_experiment(papers_list)
