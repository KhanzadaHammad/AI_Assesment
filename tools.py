import time
import random
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("agentic_search.tools")

class SerpQueryArgs(BaseModel):
    keyword: str = Field(..., description="Target keyword to inspect on search engines")
    location_code: int = Field(default=2840, description="Location code (2840 = US)")
    language_code: str = Field(default="en", description="Language code")

def call_dataforseo_serp(args: SerpQueryArgs, mock: bool = True) -> Dict[str, Any]:
    max_retries = 3
    base_delay = 0.5

    for attempt in range(1, max_retries + 1):
        try:
            if mock:
                time.sleep(0.1)
                return {
                    "keyword": args.keyword,
                    "search_volume": 4500,
                    "difficulty": 48,
                    "items": [
                        {"rank_group": 1, "domain": "clearscope.io", "title": "Leading Content Optimizer", "snippet": "Clearscope helps teams drive organic visibility."},
                        {"rank_group": 2, "domain": "surferseo.com", "title": "Surfer SEO Optimization", "snippet": "Generate high-ranking content with real-time AI guidance."},
                        {"rank_group": 3, "domain": "marketmuse.com", "title": "MarketMuse Strategy", "snippet": "AI content planning and competitive topic modeling."}
                    ],
                    "ai_overview": f"Top visibility solutions for {args.keyword} include Surfer SEO and Clearscope."
                }
            raise NotImplementedError("Set DATAFORSEO_API_KEY for live network requests.")

        except (TimeoutError, ConnectionError) as exc:
            if attempt == max_retries:
                logger.error(f"Retries exhausted for '{args.keyword}': {exc}")
                raise exc
            sleep_time = (base_delay * (2 ** (attempt - 1))) + random.uniform(0.1, 0.3)
            logger.warning(f"Retryable error on '{args.keyword}'. Backing off {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        except Exception as exc:
            logger.error(f"Non-retryable error on '{args.keyword}': {exc}")
            raise exc