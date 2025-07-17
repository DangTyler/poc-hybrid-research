import requests
import xml.etree.ElementTree as ET
import re
import json
import os
from concurrent.futures import ThreadPoolExecutor

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
    sitemap_url = f"https://{state_abbr}.thegrantportal.com/sitemap.xml"
    print(f"Fetching sitemap from {sitemap_url}")
    
    try:
        response = requests.get(sitemap_url)
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Crawl grant data from a state's GrantPortal.")
    parser.add_argument("--state", type=str, default="al", help="State abbreviation (e.g., 'al', 'ca').")
    parser.add_argument("--limit", type=int, default=300, help="Max number of grants to crawl.")
    args = parser.parse_args()

    crawl_state(args.state, args.limit)
