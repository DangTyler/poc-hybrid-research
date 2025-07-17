# Hybrid Pinecone + Neo4j PoC (8-Hour Sprint)
## Goal
Deliver a live Streamlit site that:
1. Finds the **most-cited** research paper on a chosen topic (Semantic Scholar slice).
2. Answers **multi-constraint grant queries** on one GrantPortal state slice.
3. Showcases Neo4j LLM Graph Builder quality & hybrid retrieval latency.

---

## 0. Repo Setup
☐ `git init poc-hybrid-research`  
☐ `python -m venv .venv && source .venv/bin/activate`  
☐ `pip install neo4j pinecone-client openai python-dotenv pandas streamlit tiktoken`  
☐ Create project tree  

poc-hybrid-research/
│ .env.example
│ requirements.txt
│ README.md
├─ data/
├─ ingest/
│ ├─ papers_ingest.py
│ └─ grants_ingest.py
├─ retrieval/
│ ├─ hybrid_query.py
│ └─ tests_smoke.py
└─ app/
└─ demo.py

yaml
Copy
Edit

---

## 1. Semantic Scholar ► Pinecone ► Neo4j (Research Pipeline)
### 1-A Fetch & Stage
☐ **cursor:** “Generate `papers_ingest.py` with `fetch_papers(topic, limit=100)` pulling `paperId,title,abstract,citationCount,references` via Semantic Scholar Graph API; save to `data/papers.json`.”

### 1-B Load to Neo4j Aura Free
☐ **cursor:** “In `papers_ingest.py`, add `load_to_neo4j(papers)` that MERGEs `(:Paper)` nodes & optional `[:CITES]` edges; set `citationCount` prop.”

### 1-C Build Pinecone Index
☐ **cursor:** “Add `load_to_pinecone(papers)`—compute OpenAI `text-embedding-3-small` for `title + abstract`, upsert with `metadata.graph_id = "Paper:<id>"`.”

---

## 2. GrantPortal Slice ► KG-Builder
### 2-A Crawl One State
☐ **cursor:** “Create `grants_ingest.py` with `crawl_state(state_abbr)`, pulling ~300 grant pages from `https://<state>.thegrantportal.com/sitemap.xml`; extract `id,title,amount,state,topic,deadline`.”

### 2-B KG-Builder GUI
☐ Manually upload 5 sample grant pages to **Neo4j LLM Graph Builder web GUI**  
☐ Lock ontology: `Grant, Topic, State, DEADLINE` + edges `FOCUS_ON, ELIGIBLE_FOR, DEADLINE`  
☐ Export failures list for QC.

### 2-C Scripted GraphRAG Extraction
☐ **cursor:** “In `grants_ingest.py`, use GraphRAG `KGBuild()` to parse remaining pages with same ontology; ingest to Aura.”

☐ Upsert grant description embeddings to Pinecone with `graph_id = "Grant:<id>"`.

---

## 3. Hybrid Retrieval Core
☐ **cursor:** “Create `retrieval/hybrid_query.py` with `query(prompt, domain="papers"|"grants")`:

1. Embed prompt → Pinecone `top_k=15`
2. Extract `graph_id`s → Cypher template  
   • papers: rank by `citationCount`  
   • grants: filter `amount_max<=500000 AND state='AL'` + sort by deadline
3. Return dict + latency ms”

☐ **cursor:** “Write `tests_smoke.py` verifying:
* research: top doc citationCount == max(json)  
* grants: matches hard-coded ground truth for 3 sample queries”

---

## 4. Streamlit Web App
☐ **cursor:** “Generate `app/demo.py`:

* Sidebar toggle (Research / Grants)  
* Text input → `hybrid_query.query()`  
* Show title, metric, and Cypher path  
* Display latency”

☐ Add `.streamlit/secrets.toml` for keys (or load `.env`).

---

## 5. Deployment
☐ Push to GitHub (public or private)  
☐ Render.com → New → Python/Streamlit service  
  • Add env vars (`OPENAI_API_KEY`, `PINECONE_KEY`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`)  
☐ Confirm public URL works.

---

## 6. Validation & Insights
☐ Run `python retrieval/tests_smoke.py` — ensure ✅  
☐ Gather metrics:

| Dataset | Nodes | Edges | Top-k accuracy | p95 latency |
|---------|-------|-------|----------------|-------------|
| Papers (100) | ___ | ___ | 100 % | ___ ms |
| Grants (300) | ___ | ___ | ___ % | ___ ms |

☐ Manual QC: sample 20 triples; note error rate & schema violations.

---

## 7. Demo Materials (Google Meet)
☐ `slides/PoC_demo.pptx` with 5 slides:  
1. Objective & tech stack  
2. Live URL screenshot (Streamlit)  
3. Metrics table (above)  
4. Screenshot of Neo4j LLM Graph Builder showing ontology & nodes  
5. Next-step roadmap (scale, biases, cost)

☐ Prepare 2-minute narration:

* **Hybrid win (papers):** vector recall + graph ranking correct every time.  
* **KG-Builder quality:** <X % invalid triples after ontology lock.  
* **Latency:** sub-1 s end-to-end.  
* **Business relevance:** same pipeline solves grant eligibility joins.

---

## 8. Stretch Goals (if time)
☐ Add **`.gradio`** alternative UI  
☐ Integrate **citation path** (graph visualization) with `pyvis`  
☐ Batch ingest 10 k grants to test Aura limits  
☐ Cost projection slide for scaling

---

## ENV Checklist before commit
☐ `.env.example` updated with keys placeholders  
☐ `requirements.txt` frozen (`pip freeze > requirements.txt`)  
☐ `README.md` instructions: `python ingest/papers_ingest.py --topic "graph neural networks"` etc.

---

*Paste this file into Cursor.  
Work top-to-bottom; mark checkboxes as you complete or let Cursor generate each code block.* 
3-Line Insight Script for the Demo
“Vector search gives us fuzzy recall; the graph layer then counts or joins — that’s how we nail ‘most-cited paper’ and ‘STEM grants < $500 K in Alabama’.
Neo4j LLM KG-Builder auto-created schema-valid triples with < 4 % rejection rate after ontology lock.
The whole answer loop runs in about a second, fully explainable via the Cypher path we return.”

Memorize / tweak those lines, open the Streamlit URL, and you’re ready to impress. 