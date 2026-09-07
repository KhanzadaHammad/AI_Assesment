import uuid
from typing import Dict, Any, Optional, List

class InMemoryStore:
    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.runs: Dict[str, List[Dict[str, Any]]] = {}
        self.queries: Dict[str, List[Dict[str, Any]]] = {}
        self.recommendations: Dict[str, List[Dict[str, Any]]] = {}

    def create_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        p_id = str(uuid.uuid4())
        record = {
            "profile_uuid": p_id,
            "name": data["name"],
            "domain": data["domain"],
            "industry": data.get("industry", ""),
            "description": data.get("description", ""),
            "competitors": data.get("competitors", []),
            "status": "created"
        }
        self.profiles[p_id] = record
        self.runs[p_id] = []
        self.queries[p_id] = []
        self.recommendations[p_id] = []
        return record

    def get_profile(self, p_id: str) -> Optional[Dict[str, Any]]:
        return self.profiles.get(p_id)

db = InMemoryStore()