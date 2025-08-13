import streamlit as st
import sys
import os
import json
import pandas as pd

# Add project root to allow importing retrieval package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval.hybrid_query import query as hybrid_query, get_co_citations, driver as neo4j_driver
from retrieval.graph_qa import answer_question as graph_qa
from ingest.upload_ingest import ingest_uploaded_docs

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hybrid Search PoC", layout="wide", initial_sidebar_state="expanded")

# --- HELPER FUNCTIONS ---
def get_llm_experiment_results():
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)
            WHERE p.llm_extracted_citations IS NOT NULL
            RETURN 
                p.title AS title, 
                p.citationCount AS ground_truth, 
                p.llm_extracted_citations AS llm_extracted
        """)
        return [record.data() for record in result]

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 Hybrid Search PoC")
    st.header("How the Knowledge Graph was Built")
    
    with st.expander("📄 Papers KG: Structured Ingestion"):
        st.markdown("A Python script mapped structured API data to a graph schema: `(Paper)-[:CITES]->(Paper)`. This is our reliable **ground truth**.")
    
    # LLM Builder experiment copy removed to avoid over-claiming accuracy

    with st.expander("📊 Graph Queries vs. Graph Analytics"):
        st.markdown("""
        **This PoC uses real-time Graph Queries:** The demos use Cypher to instantly traverse relationships, filter results, and find complex paths. This is ideal for user-facing applications.
        
        **It does NOT use Graph Analytics:** We do not run algorithms like PageRank or Community Detection across the whole dataset. These are typically run offline and are a powerful next step for enriching the graph with new insights (e.g., creating an `authorityScore` property), but are not part of this real-time demo.
        """)

    st.markdown("---")
    st.header("Demo Scenarios")
    
    app_mode = st.radio(
        "Choose a demo scenario:",
        ("LLM Builder Accuracy", "Relationship Discovery", "Natural Language Q&A", "User Upload (Insight Mode)")
    )

# --- MAIN APP ---

# Scenario 0: LLM Builder Accuracy
if app_mode == "LLM Builder Accuracy":
    st.header("LLM Builder Accuracy Test")
    st.markdown("We test the LLM's ability to extract a specific number (`citationCount`) from unstructured text and compare against ground truth from the API.")
    
    with st.expander("How the Graph is Used Here"):
        st.markdown("""
        1.  **Storing Ground Truth:** The graph holds both the paper's abstract text and the true `citationCount` from a reliable API.
        2.  **Storing LLM Output:** We use an LLM to read the abstract and extract the citation number, storing its answer as a new property (`llm_extracted_citations`) on the `Paper` node.
        3.  **Direct Comparison:** A simple Cypher query then pulls both properties from the *same node*, allowing for a direct, credible accuracy measurement. This validates the LLM's ability to act as a reliable graph builder.
        """)
    
    st.code("""
    You are a helpful assistant. Given the following text from a research paper abstract, extract the exact integer value for the number of citations the paper has.
    
    Only return the integer. If the citation count is not mentioned, return 0.
    
    Abstract:
    {abstract_text}
    """, language="markdown")

    results = get_llm_experiment_results()
    if not results:
        st.warning("LLM experiment data not found. Please run the ingestion script with the `--run-llm-experiment` flag.")
    else:
        df = pd.DataFrame(results)
        df['correct'] = df['ground_truth'] == df['llm_extracted']
        accuracy = (df['correct'].sum() / len(df)) * 100
        
        st.info(f"Observed LLM extraction accuracy: **{accuracy:.2f}%** (not 100%).")
        st.markdown("Parsing and prompting matter. PDFs/references need reliable parsing; tighter prompts and validations reduce off‑by‑N errors.")
        
        with st.expander("Show Raw LLM Experiment Data from Graph"):
            st.json(results)
            
        st.dataframe(df)

# Scenario 1: Contextual Ranking
elif app_mode == "Contextual Ranking":
    st.header("🎯 Act 2: Contextual Ranking (Papers)")
    st.markdown("This demo shows how a graph provides **contextual ranking** that a pure vector search cannot.")
    
    with st.expander("How the Graph is Used Here"):
        st.markdown("""
        1.  **Vector Search First:** The user's query is sent to Pinecone, which returns a list of papers based on semantic similarity (the `vector_results`). The LLM (`text-embedding-3-small`) is used here to turn the user's query and the paper abstracts into vectors.
        2.  **Graph Filtering:** The IDs of these papers are sent to Neo4j. We fetch a critical piece of context that the vector DB doesn't have: the `citationCount`.
        3.  **Intelligent Re-Ranking:** We compute a `hybrid_score` that combines the vector similarity score with the paper's citation count. The final `hybrid_results` are sorted by this score, pushing more authoritative papers to the top.
        """)
        st.info("""
        **Beyond Simple Ranking:** You're right, you *could* just add `citationCount` to the vector metadata for this simple case. 
        
        The real power comes from using graph algorithms for ranking. Imagine a score based on a paper's **PageRank** or **centrality** *within the context of the search results*. This measures a paper's authority and connectedness in a way that is impossible if the data is just flat metadata. That is the true, defensible value of the graph in contextual ranking.
        """)

    prompt = st.text_input("Enter a research topic:", "graph neural networks")

    if st.button("Run Hybrid Search", key="papers"):
        response = hybrid_query(prompt, domain="papers")
        st.success(f"Query completed in {response['latency_ms']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Vector-Only Results")
            st.markdown("Ranked by semantic similarity. The most cited paper isn't #1.")
            with st.expander("Show Raw Vector-Only Data"):
                st.json(response['vector_results'])
        
        with col2:
            st.subheader("Hybrid Results (Graph-Powered)")
            st.markdown("Re-ranked by **Hybrid Score** (`vector_score * log(citationCount)`).")
            with st.expander("Show Raw Hybrid Data"):
                st.json(response['hybrid_results'])
            
            for res in response['hybrid_results']:
                st.markdown(f"**{res['title']}**")
                st.markdown(f"Citation Count: {res.get('citationCount', 0)} | Hybrid Score: {res.get('hybrid_score', 'N/A')}")
                if res.get('url'):
                    st.markdown(f"[Verify Source]({res['url']})")
                st.divider()

# Scenario 2: Multi-Constraint Filtering
elif app_mode == "Multi-Constraint Filtering":
    st.header("💰 Act 3: Multi-Constraint Filtering (Grants)")
    st.markdown("This demo shows how a graph applies **multiple hard filters** impossible for vector search.")
    
    with st.expander("How the Graph is Used Here"):
        st.markdown("""
        1.  **Vector Search First:** The query finds grants in Pinecone based on the text description. The results, however, contain grants from all states and funding amounts.
        2.  **Precise Graph Filtering:** We take the IDs from the vector search and run a Cypher query in Neo4j. This query applies three specific constraints:
            *   `MATCH (g:Grant)-[:ELIGIBLE_FOR]->(s:State {name: 'AL'})` - Must be in Alabama.
            *   `WHERE g.amount <= 500000` - Must be under $500,000.
            *   It also traverses the graph to fetch the `Deadline` for sorting.
        3.  **Verifiable Results:** The final `hybrid_results` contain only the grants that match all criteria, a task that is impossible for a vector-only approach.
        """)
        st.info("""
        **The Crucial Difference: Properties vs. Relationships**
        
        You are right to ask: "Couldn't I just add `state` and `amount` as metadata to my vector DB?" Yes, you could. 
        
        But that only allows you to filter on the **properties** of a single object. The unique power of the graph is filtering based on **relationships**. Our query `MATCH (g:Grant)-[:ELIGIBLE_FOR]->(s:State {name: 'AL'})` is not just checking a property; it is **traversing a connection** in the graph.
        
        This is a simple example, but it represents a fundamentally different capability. It unlocks the ability to ask questions like, "Find grants from organizations that have previously funded projects that are connected to the topic of 'Education'," which requires traversing multiple relationship hops and is impossible with a metadata filter. **Act 4 is a perfect example of this unique power.**
        """)

    st.info("""
    **Data Provenance Note:** The grant data for this demo is from the static file `data/grants_al.json`. 
    This is because the live grant portal blocked our crawler. This section demonstrates the *filtering logic* using a small, controlled dataset.
    """, icon="ℹ️")

    with st.expander("Show `data/grants_al.json` - The Source of Truth for this Demo"):
        with open("data/grants_al.json", "r") as f:
            st.json(json.load(f))

    prompt = st.text_input("Enter a grant topic:", "STEM programs")

    if st.button("Run Hybrid Search", key="grants"):
        response = hybrid_query(prompt, domain="grants")
        st.success(f"Query completed in {response['latency_ms']}")
        
        st.info("**Cypher Query Used:**")
        st.code(response['cypher_query'], language='cypher')

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Vector-Only Results")
            st.markdown("Contains irrelevant results from other states and with incorrect amounts.")
            with st.expander("Show Raw Vector-Only Data"):
                st.json(response['vector_results'])
            
            for res in response['vector_results']:
                st.markdown(f"**{res['title']}**")
                st.markdown(f"Amount: `${res.get('amount', 0):,}` | State: {res.get('state', 'N/A')}")
                st.divider()
        
        with col2:
            st.subheader("Hybrid Results (Graph-Powered)")
            st.markdown("Correctly filtered to **only** grants in **Alabama** under **$500,000**.")
            
            with st.expander("Show Raw Graph DB Data"):
                st.json(response['hybrid_results'])

            for res in response['hybrid_results']:
                st.markdown(f"**{res['title']}**")
                st.markdown(f"Amount: `${res.get('amount', 0):,}` | Deadline: {res.get('deadline', 'N/A')}")
                if res.get('url'):
                    st.markdown(f"[Verify Source]({res['url']})")
                st.divider()

# Scenario 3: Relationship Discovery
elif app_mode == "Relationship Discovery":
    st.header("Relationship Discovery (Co-Citation)")
    st.markdown("This demo showcases a query that **only a graph can answer**: finding hidden connections.")
    
    with st.expander("How the Graph is Used Here"):
        st.markdown("We run the pattern `(p1)-[:CITES]->(c)<-[:CITES]-(p2)` to find works co‑cited by the two top papers (one per topic). On small datasets or uploads without deterministic citations, this can return no results.")

    topic1 = st.text_input("Enter the first research topic:", "comprehensive survey gnn")
    topic2 = st.text_input("Enter the second research topic:", "benchmarking graph neural networks")

    if st.button("Find Foundational Papers", key="cocite"):
        with st.container():
            st.subheader("Vector-Only Approach")
            st.error("Impossible. A vector database cannot analyze paths between two distinct query results.")

        with st.container():
            st.subheader("Graph-Powered Discovery")
            response = get_co_citations(topic1, topic2)
            
            st.info("**Cypher Query Used:**")
            st.code(response['cypher_query'], language='cypher')
            
            st.markdown(f"Found **{len(response['results'])}** co‑cited papers (if any):")
            with st.expander("Show Co-Citation Results"):
                st.json(response.get('results', []))

elif app_mode == "Natural Language Q&A":
    st.header("💬 Natural Language Q&A (GraphCypherQAChain)")
    st.markdown("Ask a question in natural language. The system will generate Cypher, execute it on Neo4j, and return the answer.")

    # Optional dataset scoping
    with neo4j_driver.session() as session:
        datasets = [r[0] for r in session.run("MATCH (n) WHERE n.dataset IS NOT NULL RETURN DISTINCT n.dataset ORDER BY n.dataset LIMIT 50")] 
    dataset_choice = st.selectbox("Scope to dataset (optional)", options=["All"] + datasets, index=0)

    # Quick-test buttons for CEO
    colq1, colq2, colq3 = st.columns(3)
    with colq1:
        if st.button("Top 5 GNN papers", key="btn_top5"):
            st.session_state["qa_prefill"] = "Top 5 papers on 'graph neural networks' with links"
    with colq2:
        if st.button("Cite → 'How Powerful are GNNs?'", key="btn_cite_to"):
            st.session_state["qa_prefill"] = "Which papers cite 'How Powerful are Graph Neural Networks?'"
    with colq3:
        if st.button("Cited by ← 'How Powerful are GNNs?'", key="btn_cited_by"):
            st.session_state["qa_prefill"] = "Which papers are cited by 'How Powerful are Graph Neural Networks?'"

    # List all papers for the selected dataset
    if st.button("List all papers (dataset)", key="btn_list_all"):
        ds = None if dataset_choice == "All" else dataset_choice
        if ds:
            cy = (
                "MATCH (p:Paper {dataset:$ds})\n"
                "WHERE coalesce(p.stub,false)=false\n"
                "RETURN p.title AS title, p.citationCount AS citationCount, p.url AS url\n"
                "ORDER BY citationCount DESC"
            )
            params = {"ds": ds}
        else:
            cy = (
                "MATCH (p:Paper)\n"
                "WHERE coalesce(p.stub,false)=false\n"
                "RETURN p.title AS title, p.citationCount AS citationCount, p.url AS url\n"
                "ORDER BY citationCount DESC"
            )
            params = {}
        with neo4j_driver.session() as session:
            rows = [r.data() for r in session.run(cy, **params)]
        st.caption("Cypher used:")
        st.code(cy, language="cypher")
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No papers found for the current dataset scope.")

    default_q = st.session_state.get("qa_prefill", "Which papers have the highest citationCount?")
    question = st.text_input("Your question:", default_q)
    if st.button("Ask", key="qa"):
        try:
            scoped_q = question
            if dataset_choice and dataset_choice != "All":
                scoped_q = f"Only use dataset '{dataset_choice}'. " + question
            result = graph_qa(scoped_q)
            st.success(result.get("answer"))
            with st.expander("Show generated Cypher and raw output"):
                st.code(result.get("cypher", ""), language="cypher")
                st.json(result.get("raw", {}))
        except Exception as e:
            st.error(f"Error: {e}")

    # Visualization temporarily disabled for this demo

elif app_mode == "User Upload (Insight Mode)":
    st.header("📤 Upload documents → infer topics/relations (no hard-coded topic)")
    st.markdown("Files are embedded immediately and optionally enriched with the LLM builder. All nodes/edges are tagged with a dataset ID.")

    dataset = st.text_input("Dataset ID", value="demo_upload")
    run_builder = st.checkbox("Run LLM Graph Builder (topics/authors/links)", value=True)
    run_pinecone = st.checkbox("Embed to Pinecone for search", value=True)
    files = st.file_uploader("Upload .pdf/.txt/.md", accept_multiple_files=True)

    if st.button("Ingest Uploads"):
        res = ingest_uploaded_docs(files or [], dataset=dataset, run_builder=run_builder, run_pinecone=run_pinecone)
        st.success(f"Ingested {res.get('ingested', 0)} documents into dataset '{dataset}'.")
