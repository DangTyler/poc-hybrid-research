import requests
import json
import os
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv
import pinecone
from openai import OpenAI

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
                'references': references
            }
            papers.append(paper)
    
    if not os.path.exists('data'):
        os.makedirs('data')

    with open('data/papers.json', 'w') as f:
        json.dump(papers, f, indent=2)
    
    print(f"Successfully saved {len(papers)} papers to data/papers.json")
    return papers

def load_to_neo4j(papers):
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
        # Create papers
        for paper in papers:
            session.run("""
                MERGE (p:Paper {paperId: $paperId})
                SET p.title = $title, p.citationCount = $citationCount
            """, paperId=paper['paperId'], title=paper['title'], citationCount=paper['citationCount'])

        # Create citation relationships
        for paper in papers:
            if paper.get('references'):
                for ref_id in paper['references']:
                    # Ensure the referenced paper exists in our dataset to avoid dead-end relationships
                    if any(p['paperId'] == ref_id for p in papers):
                        session.run("""
                            MATCH (p1:Paper {paperId: $p1_id})
                            MATCH (p2:Paper {paperId: $p2_id})
                            MERGE (p1)-[:CITES]->(p2)
                        """, p1_id=paper['paperId'], p2_id=ref_id)
        
        print(f"Successfully loaded {len(papers)} papers and their citations into Neo4j.")

    driver.close()

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
    for paper in papers:
        if paper.get('abstract'):
            text_to_embed = f"{paper['title']}: {paper['abstract']}"
            
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Fetch papers from Semantic Scholar and load to Neo4j and Pinecone.")
    parser.add_argument("--topic", type=str, default="graph neural networks", help="Topic to search for papers on.")
    parser.add_argument("--limit", type=int, default=100, help="Number of papers to fetch.")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching and only load from existing json file.")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip loading to Neo4j.")
    parser.add_argument("--skip-pinecone", action="store_true", help="Skip loading to Pinecone.")
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
        if not args.skip_neo4j:
            load_to_neo4j(papers_list)
        if not args.skip_pinecone:
            load_to_pinecone(papers_list)
