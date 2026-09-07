import uuid
import logging
from typing import Dict, Any, List, TypedDict
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, END

from config import get_llm, get_embedder
from tools import SerpQueryArgs, call_dataforseo_serp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agentic_search.graph")

class SearchIntelligenceState(TypedDict):
    profile_uuid: str
    target_domain: str
    competitors: List[str]
    user_query: str
    planned_queries: List[str]
    retrieval_results: List[Dict[str, Any]]
    normalized_data: List[Dict[str, Any]]
    rag_context: str
    analysis_insights: List[str]
    recommendations: List[Dict[str, Any]]
    final_report: Dict[str, Any]
    status: str

class PlannedSearchQueries(BaseModel):
    queries: List[str] = Field(
        ...,
        description="2 to 3 targeted search query strings to assess AI visibility and ranking gaps"
    )

# 1. Query Planner Agent (LLM)
def query_planner_node(state: SearchIntelligenceState) -> Dict[str, Any]:
    logger.info("Executing Query Planner Agent (LLM)")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an SEO search planner. Return 2-3 target search keywords to inspect visibility for {domain}."),
        ("user", "User question: {user_query}")
    ])
    
    chain = prompt | llm.with_structured_output(PlannedSearchQueries)
    try:
        plan: PlannedSearchQueries = chain.invoke({
            "domain": state["target_domain"],
            "user_query": state["user_query"]
        })
        queries = plan.queries
    except Exception as exc:
        logger.warning(f"Fallback planning triggered: {exc}")
        queries = [
            state["user_query"],
            f"best tools alternative to {state['target_domain']}"
        ]
    return {"planned_queries": queries}

# 2. Search / Retrieval Agent (Tools)
def retrieval_node(state: SearchIntelligenceState) -> Dict[str, Any]:
    logger.info("Executing Search/Retrieval Agent")
    results = []
    has_partial_failure = False

    for query in state["planned_queries"]:
        try:
            args = SerpQueryArgs(keyword=query)
            data = call_dataforseo_serp(args, mock=True)
            results.append({"status": "success", "data": data, "keyword": query})
        except Exception as exc:
            logger.error(f"Retrieval error for '{query}': {exc}")
            has_partial_failure = True
            results.append({"status": "failed", "error": str(exc), "keyword": query})

    return {
        "retrieval_results": results,
        "status": "partial" if has_partial_failure else "retrieved"
    }

def retrieval_routing(state: SearchIntelligenceState) -> str:
    return "fallback" if state["status"] == "partial" else "normalizer"

def fallback_node(state: SearchIntelligenceState) -> Dict[str, Any]:
    logger.warning("Routing through Fallback Node due to partial retrieval failures.")
    return {"status": "degraded_recovery"}

# 3. Extraction / Normalization Agent
def normalizer_node(state: SearchIntelligenceState) -> Dict[str, Any]:
    logger.info("Executing Extraction/Normalization Agent")
    normalized = []
    target = state["target_domain"].lower()

    for entry in state["retrieval_results"]:
        if entry.get("status") == "failed":
            continue

        raw = entry["data"]
        position = None
        for item in raw.get("items", []):
            if target in item.get("domain", "").lower():
                position = item.get("rank_group")
                break

        normalized.append({
            "query_uuid": str(uuid.uuid4()),
            "query_text": raw["keyword"],
            "search_volume": raw.get("search_volume", 1000),
            "difficulty": raw.get("difficulty", 50),
            "domain_visible": position is not None,
            "visibility_position": position,
            "raw_items": raw.get("items", []),
            "discovered_at": datetime.now(timezone.utc).isoformat()
        })
    return {"normalized_data": normalized}

# 4. RAG Indexer Agent (Embeddings)
def rag_indexer_node(state: SearchIntelligenceState) -> Dict[str, Any]:
    logger.info("Executing RAG Indexer Agent (Vector Embeddings)")
    embedder = get_embedder()
    docs = []

    for item in state["normalized_data"]:
        for serp in item.get("raw_items", []):
            text = f"Keyword: {item['query_text']} | Rank {serp.get('rank_group')}: {serp.get('domain')} - {serp.get('title')}. Snippet: {serp.get('snippet', '')}"
            docs.append(Document(page_content=text, metadata={"keyword": item["query_text"]}))

    if not docs:
        return {"rag_context": "No indexed snippets available."}

    vector_store = InMemoryVectorStore.from_documents(docs, embedder)
    relevant_docs = vector_store.similarity_search(state["target_domain"], k=3)
    context_text = "\n".join([doc.page_content for doc in relevant_docs])
    return {"rag_context": context_text}

# 5. Analysis / Synthesis Agent (LLM + Formula)
def analysis_node(state: SearchIntelligenceState) -> Dict[str, Any]:
    logger.info("Executing Analysis/Synthesis Agent (LLM)")
    llm = get_llm()
    recommendations = []

    for item in state["normalized_data"]:
        vol = item["search_volume"]
        diff = max(item["difficulty"], 1)
        rank = item["visibility_position"] or 100
        score = round(min(1.0, (vol / 5000.0) * (rank / 100.0) * (1 - (diff / 150.0))), 2)
        item["opportunity_score"] = max(0.05, score)

        if rank > 1:
            recommendations.append({
                "recommendation_uuid": str(uuid.uuid4()),
                "target_query_uuid": item["query_uuid"],
                "content_type": "landing_page" if "alternative" in item["query_text"] else "blog_post",
                "title": f"Action Plan for {item['query_text'].title()}",
                "rationale": f"Current rank is {rank}. Optimization will improve presence in AI Overviews.",
                "target_keywords": [item["query_text"]],
                "priority": "high" if score > 0.4 else "medium"
            })

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI visibility analyst. Synthesize competitive ranking gaps using the provided context."),
        ("user", "Brand: {domain}\nCompetitors: {competitors}\nRetrieved Context:\n{context}")
    ])
    chain = prompt | llm
    insights_response = chain.invoke({
        "domain": state["target_domain"],
        "competitors": state["competitors"],
        "context": state["rag_context"]
    })

    return {
        "analysis_insights": [str(insights_response.content)],
        "recommendations": recommendations
    }

# 6. Report Agent (LLM)
def report_node(state: SearchIntelligenceState) -> Dict[str, Any]:
    logger.info("Executing Report Agent (LLM)")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Create a concise, executive-level markdown summary of visibility findings."),
        ("user", "Findings: {insights}\nTotal Queries: {count}")
    ])
    chain = prompt | llm
    summary_msg = chain.invoke({
        "insights": state["analysis_insights"],
        "count": len(state["normalized_data"])
    })

    report = {
        "executive_summary": str(summary_msg.content),
        "total_queries_evaluated": len(state["normalized_data"]),
        "recommendations_count": len(state["recommendations"])
    }
    return {"final_report": report, "status": "completed"}

# Build LangGraph DAG
workflow = StateGraph(SearchIntelligenceState)
workflow.add_node("planner", query_planner_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("fallback", fallback_node)
workflow.add_node("normalizer", normalizer_node)
workflow.add_node("rag_indexer", rag_indexer_node)
workflow.add_node("analysis", analysis_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "retrieval")
workflow.add_conditional_edges("retrieval", retrieval_routing, {
    "fallback": "fallback",
    "normalizer": "normalizer"
})
workflow.add_edge("fallback", "normalizer")
workflow.add_edge("normalizer", "rag_indexer")
workflow.add_edge("rag_indexer", "analysis")
workflow.add_edge("analysis", "report")
workflow.add_edge("report", END)

intelligence_dag = workflow.compile()