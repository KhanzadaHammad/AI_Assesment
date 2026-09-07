from fastapi.testclient import TestClient
from app import app
from tools import SerpQueryArgs

client = TestClient(app)

def test_argument_validation():
    args = SerpQueryArgs(keyword="seo ranking tool")
    assert args.keyword == "seo ranking tool"
    assert args.location_code == 2840

def test_full_pipeline_flow():
    # 1. Create Business Profile
    res = client.post("/api/v1/profiles", json={
        "name": "Surfer SEO",
        "domain": "surferseo.com",
        "industry": "SEO Software",
        "competitors": ["clearscope.io", "marketmuse.com"]
    })
    print("Response:",res)
    assert res.status_code == 201
    profile_uuid = res.json()["profile_uuid"]

    # 2. Run DAG Pipeline (Planner -> Retrieval -> Normalizer -> RAG -> Analysis -> Report)
    run_res = client.post(f"/api/v1/profiles/{profile_uuid}/run?query=best+content+optimization+tool")
    assert run_res.status_code == 200
    run_body = run_res.json()
    assert run_body["status"] == "completed"
    assert run_body["retrieval_calls_planned"] > 0
    assert run_body["records_normalized"] > 0
    assert "executive_summary" in run_body["final_report"]

    # 3. Verify Queries and Pagination
    q_res = client.get(f"/api/v1/profiles/{profile_uuid}/queries?min_score=0.0")
    assert q_res.status_code == 200
    assert len(q_res.json()) > 0

    # 4. Verify Recommendations
    rec_res = client.get(f"/api/v1/profiles/{profile_uuid}/recommendations")
    assert rec_res.status_code == 200
    assert len(rec_res.json()) > 0

test_argument_validation()
test_full_pipeline_flow()