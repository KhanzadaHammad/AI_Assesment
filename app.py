import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel

from store import db
from graph import intelligence_dag

app = FastAPI(title="Agentic Search Intelligence API", version="2.0")

class ProfileCreate(BaseModel):
    name: str
    domain: str
    industry: Optional[str] = None
    description: Optional[str] = None
    competitors: List[str] = []

@app.post("/api/v1/profiles", status_code=status.HTTP_201_CREATED)
def create_profile(payload: ProfileCreate):
    return db.create_profile(payload.model_dump())

@app.get("/api/v1/profiles/{profile_uuid}")
def get_profile(profile_uuid: str):
    profile = db.get_profile(profile_uuid)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    queries = db.queries.get(profile_uuid, [])
    avg_score = 0.0
    if queries:
        avg_score = sum(q.get("opportunity_score", 0) for q in queries) / len(queries)

    runs = db.runs.get(profile_uuid, [])
    recent_status = runs[-1]["status"] if runs else "never_run"

    return {
        "profile": profile,
        "total_runs": len(runs),
        "most_recent_run_status": recent_status,
        "average_opportunity_score": round(avg_score, 2)
    }

@app.post("/api/v1/profiles/{profile_uuid}/run")
def trigger_dag_run(profile_uuid: str, query: str = "best project management software"):
    profile = db.get_profile(profile_uuid)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    run_uuid = str(uuid.uuid4())
    initial_state = {
        "profile_uuid": profile_uuid,
        "target_domain": profile["domain"],
        "competitors": profile.get("competitors", []),
        "user_query": query,
        "planned_queries": [],
        "retrieval_results": [],
        "normalized_data": [],
        "rag_context": "",
        "analysis_insights": [],
        "recommendations": [],
        "final_report": {},
        "status": "started"
    }

    final_state = intelligence_dag.invoke(initial_state)

    # Persist state items
    db.queries[profile_uuid] = final_state["normalized_data"]
    db.recommendations[profile_uuid] = final_state["recommendations"]

    result = {
        "pipeline_run_uuid": run_uuid,
        "status": final_state["status"],
        "retrieval_calls_planned": len(final_state["planned_queries"]),
        "records_normalized": len(final_state["normalized_data"]),
        "top_insights": final_state["analysis_insights"],
        "final_report": final_state["final_report"],
        "total_tokens_used": 0
    }
    db.runs[profile_uuid].append(result)
    return result

@app.get("/api/v1/profiles/{profile_uuid}/queries")
def get_queries(
    profile_uuid: str,
    min_score: float = Query(default=0.0),
    status: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1)
):
    items = db.queries.get(profile_uuid, [])
    
    if status == "visible":
        items = [q for q in items if q.get("domain_visible") is True]
    elif status == "not_visible":
        items = [q for q in items if q.get("domain_visible") is False]

    filtered = [q for q in items if q.get("opportunity_score", 0) >= min_score]
    filtered.sort(key=lambda x: x.get("opportunity_score", 0), reverse=True)

    start = (page - 1) * per_page
    end = start + per_page
    return filtered[start:end]

@app.get("/api/v1/profiles/{profile_uuid}/recommendations")
def get_recommendations(profile_uuid: str):
    return db.recommendations.get(profile_uuid, [])

@app.post("/api/v1/queries/{query_uuid}/recheck")
def recheck_query(query_uuid: str):
    return {
        "query_uuid": query_uuid,
        "status": "rechecked",
        "domain_visible": True,
        "visibility_position": 2
    }