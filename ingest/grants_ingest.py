import requests
import xml.etree.ElementTree as ET
import re
import json
import os
from concurrent.futures import ThreadPoolExecutor
from neo4j import GraphDatabase
from dotenv import load_dotenv
import pinecone
from openai import OpenAI

load_dotenv()

STATE_MAP = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new-hampshire",
    "nj": "new-jersey", "nm": "new-mexico", "ny": "new-york", "nc": "north-carolina",
    "nd": "north-dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode-island", "sc": "south-carolina", "sd": "south-dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west-virginia", "wi": "wisconsin", "wy": "wyoming",
}

def extract_grant_details(url):
    """
    Extracts grant details from a single grant page URL.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html = response.text

        # Use regex to find the data, this is fragile and for PoC only
        grant_id_match = re.search(r'Opportunity Number:\s*(\w+)', html)
        title_match = re.search(r'<h1 class="heading-title">(.*?)</h1>', html)
        amount_match = re.search(r'Award Ceiling:\s*\$([\d,]+)', html)
        state_match = re.search(r'Eligible Applicants:\s*.*?State governments.*?', html) # Simple check
        deadline_match = re.search(r'Current Closing Date for Applications:\s*(\w+\s\d{1,2},\s\d{4})', html)
        
        # A simple way to guess the topic from the title
        topic = "N/A"
        if title_match:
            title_text = title_match.group(1).lower()
            if "stem" in title_text or "science" in title_text:
                topic = "STEM"
            elif "health" in title_text:
                topic = "Health"
            elif "education" in title_text:
                topic = "Education"


        return {
            "id": grant_id_match.group(1) if grant_id_match else "N/A",
            "url": url,
            "title": title_match.group(1).strip() if title_match else "N/A",
            "amount": int(amount_match.group(1).replace(',', '')) if amount_match else 0,
            "state": True if state_match else False, # Simplified for PoC
            "topic": topic,
            "deadline": deadline_match.group(1) if deadline_match else "N/A"
        }
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def crawl_state(state_abbr, limit=300):
    """
    Crawls a state's GrantPortal sitemap and extracts grant information.
    """
    state_name = STATE_MAP.get(state_abbr.lower())
    if not state_name:
        print(f"Error: State abbreviation '{state_abbr}' not found in mapping.")
        return []
        
    sitemap_url = f"https://thegrantportal.com/sitemap_{state_name}_active_grants.xml"
    print(f"Fetching sitemap from {sitemap_url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(sitemap_url, headers=headers)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        
        # Namespace is often present in sitemaps
        namespace = '{http://www.sitemaps.org/schemas/sitemap/0.9}'
        grant_urls = [url.text for url in root.findall(f'.//{namespace}loc')][:limit]
        
        print(f"Found {len(grant_urls)} grant URLs. Fetching details...")
        
        grants = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(extract_grant_details, grant_urls)
            for result in results:
                if result:
                    grants.append(result)

        # Save to data/grants.json
        if not os.path.exists('data'):
            os.makedirs('data')
        with open(f'data/grants_{state_abbr}.json', 'w') as f:
            json.dump(grants, f, indent=2)
            
        print(f"Successfully saved {len(grants)} grants to data/grants_{state_abbr}.json")
        return grants

    except requests.RequestException as e:
        print(f"Failed to fetch sitemap: {e}")
        return []

def load_grants_to_neo4j_and_pinecone(grants):
    """
    Simulates the KG-Builder process by loading structured grant data into Neo4j
    and upserting their embeddings into Pinecone.
    """
    # --- Neo4j Connection ---
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, password]):
        print("Neo4j credentials not found. Skipping Neo4j load.")
        return
    driver = GraphDatabase.driver(uri, auth=(user, password))

    # --- Pinecone Connection ---
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not all([pinecone_api_key, openai_api_key]):
        print("Pinecone/OpenAI keys not found. Skipping Pinecone load.")
        return
    pc = pinecone.Pinecone(api_key=pinecone_api_key)
    openai_client = OpenAI(api_key=openai_api_key)
    index_name = "hybrid-research-poc"
    if index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(name=index_name, dimension=1536, metric="cosine", spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1'))
    index = pc.Index(index_name)

    print("Processing and loading grants...")
    with driver.session() as session:
        for grant in grants:
            # --- Neo4j Ingestion (Simulating KG-Builder) ---
            print(f"  - Loading grant '{grant['id']}' to Neo4j")
            session.run("""
                MERGE (g:Grant {id: $id})
                SET g.title = $title, g.amount = $amount, g.url = $url, g.description = $description

                MERGE (t:Topic {name: $topic})
                MERGE (s:State {name: $state})
                MERGE (d:Deadline {date: $deadline})

                MERGE (g)-[:FOCUS_ON]->(t)
                MERGE (g)-[:ELIGIBLE_FOR]->(s)
                MERGE (g)-[:HAS_DEADLINE]->(d)
            """, **grant)

            # --- Pinecone Ingestion ---
            if grant.get('description'):
                print(f"  - Embedding and upserting grant '{grant['id']}' to Pinecone")
                try:
                    res = openai_client.embeddings.create(input=[grant['description']], model="text-embedding-3-small")
                    embedding = res.data[0].embedding
                    index.upsert(vectors=[{
                        "id": grant['id'],
                        "values": embedding,
                        "metadata": {"graph_id": f"Grant:{grant['id']}", "text": grant['description']}
                    }])
                except Exception as e:
                    print(f"    Failed to process grant {grant['id']} for Pinecone: {e}")

    driver.close()
    print("Successfully loaded grants into Neo4j and Pinecone.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Crawl and ingest grant data.")
    parser.add_argument("--state", type=str, default="al", help="State abbreviation (e.g., 'al', 'ca').")
    parser.add_argument("--limit", type=int, default=300, help="Max number of grants to crawl.")
    parser.add_argument("--skip-crawl", action="store_true", help="Skip crawling and load from existing json file.")
    args = parser.parse_args()

    grant_list = []
    if not args.skip_crawl:
        grant_list = crawl_state(args.state, args.limit)
    else:
        file_path = f"data/grants_{args.state}.json"
        print(f"Skipping crawl. Loading grants from {file_path}")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                grant_list = json.load(f)
        else:
            print(f"{file_path} not found. Please run without --skip-crawl first.")

    if grant_list:
        load_grants_to_neo4j_and_pinecone(grant_list)
