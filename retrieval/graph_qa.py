import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
import re

# Reuse the deterministic co-citation pipeline when the user asks for it
from .hybrid_query import get_co_citations, driver as neo4j_driver

load_dotenv()

CYTHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=(
        "You are a Neo4j Cypher expert. Given the graph schema below, "
        "write a single read-only Cypher query that answers the user question.\n\n"
        "Schema:\n{schema}\n\n"
        "Additional hard rules you MUST follow:\n"
        "- Use ONLY these labels and relationships: Paper, Topic, Grant, State, Deadline; CITES, HAS_TOPIC, ELIGIBLE_FOR, HAS_DEADLINE, FOCUS_ON.\n"
        "- Valid properties: Paper(paperId, title, citationCount, url, llm_extracted_citations), Topic(name), Grant(id, title, amount, description, url), State(name), Deadline(date).\n"
        "- Properties and labels are case-sensitive. Do NOT invent new names.\n"
        "- Prefer numeric comparisons for citationCount and amount. Dates are ISO strings in Deadline.date.\n"
        "- Always alias returned columns to simple names (e.g., RETURN p.title AS title, p.citationCount AS citationCount).\n"
        "- Prefer excluding nulls (e.g., WHERE p.citationCount IS NOT NULL) before ordering.\n\n"
        "Question: {question}\n\nCypher query:"
    ),
)


def get_graph_qa_chain() -> GraphCypherQAChain:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        raise RuntimeError("Missing Neo4j credentials in environment")

    graph = Neo4jGraph(url=uri, username=user, password=password)
    # Ensure the chain sees current labels/properties
    graph.refresh_schema()
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        cypher_prompt=CYTHER_PROMPT,
        verbose=False,
        return_intermediate_steps=True,
        top_k=10,
        allow_dangerous_requests=True,
    )
    return chain

def answer_question(question: str):
    # Normalize smart quotes → straight quotes for robust regex
    normalized_q = (
        question
        # normalize quotes
        .replace("“", "'").replace("”", "'")
        .replace('“', "'").replace('”', "'")
        .replace('"', "'")
        # normalize hyphen-like characters to ASCII '-'
        .replace("‑", "-")  # non-breaking hyphen
        .replace("–", "-")  # en dash
        .replace("—", "-")  # em dash
    )

    # Route co-citation questions to the deterministic flow (vector candidates + Cypher)
    pattern = re.compile(r"co-?cited.*top paper for '([^']+?)'.*top paper for '([^']+?)'", re.IGNORECASE)
    m = pattern.search(normalized_q)
    if m:
        topic1, topic2 = m.group(1), m.group(2)
        res = get_co_citations(topic1, topic2)
        papers = res.get("results", [])
        if papers:
            lines = [f"- {p['title']} (citations: {p.get('citationCount', 0)})" for p in papers]
            return {"answer": "\n".join(lines), "cypher": res.get("cypher_query", ""), "raw": res}
        return {"answer": "No co-cited papers found.", "cypher": res.get("cypher_query", ""), "raw": res}

    # Top topics intents (e.g., "top topics", "top 5 topics with counts")
    m_top_n = re.search(r"top\s+(\d+)\s+topics", normalized_q, re.IGNORECASE)
    if m_top_n:
        n = int(m_top_n.group(1))
        cy = (
            "MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)\n"
            "WHERE t.name IS NOT NULL\n"
            "RETURN t.name AS topic, COUNT(p) AS count\n"
            "ORDER BY count DESC LIMIT $n"
        )
        with neo4j_driver.session() as session:
            rows = [r.data() for r in session.run(cy, n=n)]
        if rows:
            return {"answer": "\n".join(f"- {r['topic']} ({r['count']})" for r in rows), "cypher": cy, "raw": rows}
        return {"answer": "No results found in the current graph for this query.", "cypher": cy, "raw": rows}

    if re.search(r"top\s+topics", normalized_q, re.IGNORECASE) or re.search(r"topics\s+with\s+counts", normalized_q, re.IGNORECASE):
        cy = (
            "MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)\n"
            "WHERE t.name IS NOT NULL\n"
            "RETURN t.name AS topic, COUNT(p) AS count\n"
            "ORDER BY count DESC LIMIT 10"
        )
        with neo4j_driver.session() as session:
            rows = [r.data() for r in session.run(cy)]
        if rows:
            return {"answer": "\n".join(f"- {r['topic']} ({r['count']})" for r in rows), "cypher": cy, "raw": rows}
        return {"answer": "No results found in the current graph for this query.", "cypher": cy, "raw": rows}

    # Title-based fallbacks: topics connected to a paper
    m_topics = re.search(r"topics?\s+connected\s+to\s+'([^']+)'", normalized_q, re.IGNORECASE)
    if m_topics:
        title = m_topics.group(1)
        cy = (
            "MATCH (p:Paper) WHERE toLower(p.title) CONTAINS toLower($title)\n"
            "MATCH (p)-[:HAS_TOPIC]->(t:Topic)\n"
            "WHERE t.name IS NOT NULL\n"
            "RETURN DISTINCT t.name AS topic LIMIT 20"
        )
        with neo4j_driver.session() as session:
            rows = [r.data() for r in session.run(cy, title=title)]
        if rows:
            return {"answer": "\n".join(f"- {r.get('topic','(unknown)')}" for r in rows), "cypher": cy, "raw": rows}

    # Title-based fallbacks: which papers cite X / are cited by X
    m_cite = re.search(r"which\s+papers\s+cite\s+'([^']+)'", normalized_q, re.IGNORECASE)
    if m_cite:
        title = m_cite.group(1)
        cy = (
            "MATCH (p:Paper) WHERE toLower(p.title) CONTAINS toLower($title)\n"
            "MATCH (citing:Paper)-[r:CITES]->(p)\n"
            "WHERE coalesce(citing.stub,false) = false AND citing.title IS NOT NULL AND citing.citationCount IS NOT NULL\n"
            "RETURN citing.title AS title, citing.citationCount AS citationCount\n"
            "ORDER BY citationCount DESC LIMIT 10"
        )
        with neo4j_driver.session() as session:
            rows = [r.data() for r in session.run(cy, title=title)]
        if rows:
            safe = [r for r in rows if r.get('title')]
            if safe:
                return {"answer": "\n".join(f"- {r['title']} ({r.get('citationCount','N/A')})" for r in safe), "cypher": cy, "raw": safe}

    m_cited_by = re.search(r"which\s+papers\s+are\s+cited\s+by\s+'([^']+)'", normalized_q, re.IGNORECASE)
    if m_cited_by:
        title = m_cited_by.group(1)
        cy = (
            "MATCH (p:Paper) WHERE toLower(p.title) CONTAINS toLower($title)\n"
            "MATCH (p)-[r:CITES]->(cited:Paper)\n"
            "WHERE coalesce(cited.stub,false) = false AND cited.title IS NOT NULL AND cited.citationCount IS NOT NULL\n"
            "RETURN cited.title AS title, cited.citationCount AS citationCount\n"
            "ORDER BY citationCount DESC LIMIT 10"
        )
        with neo4j_driver.session() as session:
            rows = [r.data() for r in session.run(cy, title=title)]
        if rows:
            safe = [r for r in rows if r.get('title')]
            if safe:
                return {"answer": "\n".join(f"- {r['title']} ({r.get('citationCount','N/A')})" for r in safe), "cypher": cy, "raw": safe}

    chain = get_graph_qa_chain()
    result = chain.invoke({"query": question})

    # Normalize intermediate steps which can be a dict, list, or string
    cypher_text = ""
    intermediate = result.get("intermediate_steps") if isinstance(result, dict) else None
    if isinstance(intermediate, dict):
        cypher_text = intermediate.get("cypher") or intermediate.get("query") or ""
    elif isinstance(intermediate, list) and len(intermediate) > 0:
        first = intermediate[0]
        if isinstance(first, str):
            cypher_text = first
        elif isinstance(first, dict):
            cypher_text = first.get("cypher") or first.get("query") or ""
    elif isinstance(intermediate, str):
        cypher_text = intermediate

    answer_text = result.get("result") if isinstance(result, dict) else str(result)

    # Fallback: if the answer is empty/unknown but we have Cypher, try a corrected execution
    if (not answer_text) or ("don't know" in answer_text.lower()):
        cy = cypher_text or ""
        if cy.strip():
            # Normalize common patterns
            if "RETURN p.title, p.citationCount" in cy:
                cy = (
                    "MATCH (p:Paper)\n"
                    "WHERE p.citationCount IS NOT NULL\n"
                    "RETURN p.title AS title, p.citationCount AS citationCount\n"
                    "ORDER BY p.citationCount DESC\n"
                    "LIMIT 10"
                )
            # Execute
            try:
                with neo4j_driver.session() as session:
                    rows = [r.data() for r in session.run(cy)]
                # Filter out null/empty titles
                rows = [r for r in rows if r.get('title')]
                if rows:
                    formatted = [f"- {r.get('title', '')} ({r.get('citationCount', 'N/A')})" for r in rows]
                    return {"answer": "\n".join(formatted), "cypher": cy, "raw": rows}
                else:
                    # Explicitly report empty results instead of "I don't know"
                    return {"answer": "No results found in the current graph for this query.", "cypher": cy, "raw": rows}
            except Exception:
                pass

    return {
        "answer": answer_text,
        "cypher": cypher_text,
        "raw": result,
    }

