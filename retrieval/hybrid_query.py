import os
import time
import json
import math
from dotenv import load_dotenv
import pinecone
from openai import OpenAI
from neo4j import GraphDatabase

load_dotenv()

# --- Client Initializations ---
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(user, password))
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=pinecone_api_key)
index_name = "hybrid-research-poc"
index = pc.Index(index_name)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

def query(prompt, domain="papers"):
    start_time = time.time()
    prompt_embedding = openai_client.embeddings.create(input=[prompt], model="text-embedding-3-small").data[0].embedding
    
    query_response = index.query(vector=prompt_embedding, top_k=15, include_metadata=True)
    
    vector_results = []
    node_ids = []
    pinecone_scores = {}

    if query_response['matches']:
        for m in query_response['matches']:
            graph_id = m.get('metadata', {}).get('graph_id')
            if graph_id:
                node_id = graph_id.split(':')[1]
                node_ids.append(node_id)
                pinecone_scores[node_id] = m['score']
                vector_results.append({"id": m['id'], "title": graph_id, "score": m['score']})

    if not node_ids:
        return {"vector_results": [], "hybrid_results": [], "cypher_query": "No vector results found."}

    cypher_query = ""
    hybrid_results = []

    with driver.session() as session:
        if domain == "papers":
            cypher_query = """
                MATCH (p:Paper) WHERE p.paperId IN $node_ids
                RETURN p.title AS title, p.citationCount AS citationCount, p.paperId AS id, p.url as url
            """
            result = session.run(cypher_query, node_ids=node_ids)
            for record in result:
                data = record.data()
                node_id = data['id']
                vector_score = pinecone_scores.get(node_id, 0)
                citation_count = data.get('citationCount') or 0
                hybrid_score = vector_score * math.log(citation_count + 1)
                data['hybrid_score'] = f"{hybrid_score:.2f}"
                hybrid_results.append(data)
            hybrid_results = sorted(hybrid_results, key=lambda x: float(x['hybrid_score']), reverse=True)

        elif domain == "grants":
            # This is the CORRECTED logic.
            # 1. Get the raw, unfiltered details for the vector results to show the "before" state.
            # This query now correctly fetches all necessary details for the vector-only display.
            vector_only_cypher = """
                MATCH (g:Grant) WHERE g.id IN $node_ids
                OPTIONAL MATCH (g)-[:ELIGIBLE_FOR]->(s:State)
                OPTIONAL MATCH (g)-[:HAS_DEADLINE]->(d:Deadline)
                RETURN g.title as title, g.amount as amount, g.id as id, s.name as state, d.date as deadline
            """
            vector_result_records = session.run(vector_only_cypher, node_ids=node_ids)
            # We create a NEW variable for this to avoid confusion. This is what will be displayed.
            vector_results_display = [record.data() for record in vector_result_records]

            # 2. Apply the graph-based filters to get the "after" state.
            cypher_query = """
                MATCH (g:Grant)-[:ELIGIBLE_FOR]->(s:State {name: 'AL'})
                WHERE g.id IN $node_ids AND g.amount <= 500000
                WITH g
                MATCH (g)-[:HAS_DEADLINE]->(d:Deadline)
                RETURN g.title AS title, g.amount AS amount, g.id AS id, d.date as deadline, g.url as url
                ORDER BY d.date
            """
            hybrid_result_records = session.run(cypher_query, node_ids=node_ids)
            hybrid_results = [record.data() for record in hybrid_result_records]

            # 3. We now return the correct variable for the vector results.
            latency_ms = (time.time() - start_time) * 1000
            return {
                "vector_results": vector_results_display,
                "hybrid_results": hybrid_results,
                "cypher_query": cypher_query.strip(),
                "latency_ms": f"{latency_ms:.2f} ms"
            }

    # This part of the return is now only for the 'papers' domain.
    # The 'grants' domain has its own return statement.
    latency_ms = (time.time() - start_time) * 1000
    return {
        "vector_results": vector_results,
        "hybrid_results": hybrid_results,
        "cypher_query": cypher_query.strip(),
        "latency_ms": f"{latency_ms:.2f} ms"
    }

def get_co_citations(topic1, topic2):
    # Find the top paper for each topic first
    res1 = query(topic1, domain="papers")
    if not res1 or not res1['hybrid_results']:
        return {"results": [], "cypher_query": "Could not find a paper for the first topic."}
    paper1_id = res1['hybrid_results'][0]['id']

    res2 = query(topic2, domain="papers")
    if not res2 or not res2['hybrid_results']:
        return {"results": [], "cypher_query": "Could not find a paper for the second topic."}
    paper2_id = res2['hybrid_results'][0]['id']


    cypher_query = """
        MATCH (p1:Paper {paperId: $p1_id})
        MATCH (p2:Paper {paperId: $p2_id})
        MATCH (p1)-[:CITES]->(common_paper)<-[:CITES]-(p2)
        WHERE common_paper.title IS NOT NULL
        RETURN common_paper.title AS title, common_paper.citationCount AS citationCount
        ORDER BY common_paper.citationCount DESC
    """
    with driver.session() as session:
        result = session.run(cypher_query, p1_id=paper1_id, p2_id=paper2_id)
        return {
            "results": [record.data() for record in result],
            "cypher_query": cypher_query.strip()
        }
