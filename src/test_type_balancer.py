"""
Test Type Balancer - Ensures balanced mix of Knowledge vs Personality assessments

This module enforces intelligent balancing of assessment types when queries span
multiple domains (e.g., technical + behavioral requirements).

Example:
    Query: "Java developer who collaborates with teams"
    Expected: ~60% Knowledge (K) + ~40% Personality (P) tests
"""

from typing import List, Dict
import math
import logging

logger = logging.getLogger(__name__)

# Map SHL test_type values into coarse buckets
COARSE_TYPE_MAP = {
    "knowledge": "K",
    "cognitive": "K",
    "technical": "K",
    "ability": "K",  # Cognitive abilities
    "personality": "P",
    "behavioral": "P",
    "situational": "P",
    # fallback mapping - treat unknown as 'Other'
}

def coarse_type(assessment_meta: Dict) -> str:
    """
    Map assessment test_types to coarse category (K=Knowledge, P=Personality, O=Other)
    
    Args:
        assessment_meta: Dictionary with 'test_types' array or 'test_type' field
        
    Returns:
        "K", "P", or "O"
    """
    # Handle both test_types (array) and test_type (string)
    test_types = assessment_meta.get("test_types", [])
    if isinstance(test_types, str):
        test_types = [test_types]
    
    # Also check singular test_type field
    if not test_types and "test_type" in assessment_meta:
        test_types = [assessment_meta["test_type"]]
    
    # Check each test type and categorize
    for t in test_types:
        t_lower = str(t).lower()
        coarse = COARSE_TYPE_MAP.get(t_lower)
        if coarse == "K":
            return "K"  # Prioritize K if found
    
    # Check for P types
    for t in test_types:
        t_lower = str(t).lower()
        coarse = COARSE_TYPE_MAP.get(t_lower)
        if coarse == "P":
            return "P"
    
    # Default to O (Other)
    return "O"

def compute_target_allocation(total_k: int, ratios: Dict[str, float]) -> Dict[str, int]:
    """
    Convert ratio (e.g., {"K": 0.6, "P": 0.4}) into integer counts
    
    Uses greedy rounding to ensure sum equals total_k exactly.
    
    Args:
        total_k: Total number of items to allocate
        ratios: Dictionary of type -> desired proportion
        
    Returns:
        Dictionary of type -> integer count
    """
    alloc = {k: math.floor(v * total_k) for k, v in ratios.items()}
    remaining = total_k - sum(alloc.values())
    
    # Distribute remaining by largest fractional part
    frac = {k: (ratios[k] * total_k) - alloc[k] for k in ratios}
    for k, _ in sorted(frac.items(), key=lambda x: -x[1])[:remaining]:
        alloc[k] += 1
    
    return alloc

def default_ratio_from_query(enhanced_query: Dict) -> Dict[str, float]:
    """
    Infer desired K/P ratio from query content
    
    Heuristics:
    - Technical keywords (java, python, sql) -> favor Knowledge (K)
    - Collaboration keywords (team, leadership) -> add Personality (P)
    - Both -> balanced mix
    
    Args:
        enhanced_query: Dictionary with 'role', 'skills', 'level' fields
        
    Returns:
        Dictionary with keys "K" and "P" and their proportions (sum to 1.0)
    """
    text = " ".join([
        str(enhanced_query.get("role", "")),
        " ".join(enhanced_query.get("skills", [])) if enhanced_query.get("skills") else "",
        str(enhanced_query.get("level", ""))
    ]).lower()

    # Technical keywords
    has_technical = any(tok in text for tok in [
        "java", "python", "sql", "react", "spring", "node", "c++", 
        "backend", "frontend", "data", "analytics", "programming",
        "developer", "engineer", "technical", "coding", "software"
    ])
    
    # Collaboration/behavioral keywords - expanded list
    has_collab = any(tok in text for tok in [
        "team", "collaborat", "communicat", "stakeholder", "leadership",
        "manage", "manager", "coordination", "cross-functional", "interpersonal",
        "behavioral", "personality", "soft skill", "people", "relationship",
        "coordinate", "social", "emotional", "work with", "partner", "client-facing"
    ])

    # Decision logic - STRONGER ratios to force Personality items to surface
    if has_technical and has_collab:
        logger.info("Query has both technical + collaboration → 50% K / 50% P (STRONG MIX)")
        return {"K": 0.5, "P": 0.5}  # Equal mix to force P items up
    if has_technical:
        logger.info("Query is primarily technical → 80% K / 20% P")
        return {"K": 0.8, "P": 0.2}
    if has_collab:
        logger.info("Query is primarily behavioral → 30% K / 70% P")
        return {"K": 0.3, "P": 0.7}  # Strong P preference
    
    # Fallback: mostly K
    logger.info("Query unclear → default 70% K / 30% P")
    return {"K": 0.7, "P": 0.3}

def balance_candidates(
    candidates: List[Dict],
    assessment_metadata: Dict[int, Dict],
    enhanced_query: Dict = None,
    top_k: int = 10,
    custom_ratio: Dict[str, float] = None
) -> List[Dict]:
    """
    Balance candidate assessments by test type while preserving relevance order
    
    Algorithm:
    1. Categorize candidates into K/P/O buckets by test_type
    2. Sort each bucket by relevance score (descending)
    3. Allocate top_k slots according to target ratio
    4. Greedily pick from each bucket up to allocation
    5. Fill remaining slots from highest-scored candidates
    
    Args:
        candidates: List of dicts with 'id', 'score', 'name' fields
        assessment_metadata: Mapping of id -> metadata (with 'test_type')
        enhanced_query: Query info for inferring default ratio
        top_k: Number of results to return (default 10)
        custom_ratio: Optional explicit ratio (e.g., {"K": 0.5, "P": 0.5})
        
    Returns:
        Balanced list of top_k candidates, sorted by score
    """
    if not candidates:
        return []

    # Categorize candidates by coarse type
    cat_lists = {"K": [], "P": [], "O": []}
    for c in candidates:
        # Handle different ID field names
        aid = c.get("id") or c.get("assessment_id") or c.get("idx") or c.get("name")
        
        # Look up metadata by ID (which could be name or index)
        meta = assessment_metadata.get(aid, {})
        
        # If not found and we have a name, try that
        if not meta and "name" in c:
            meta = assessment_metadata.get(c["name"], {})
        
        t = coarse_type(meta)
        cat_lists.setdefault(t, []).append(c)

    # Sort each bucket by descending score
    for t in cat_lists:
        cat_lists[t].sort(key=lambda x: -float(x.get("score", x.get("hybrid_score", 0.0))))

    # Determine target allocation
    ratios = custom_ratio or default_ratio_from_query(enhanced_query or {})
    
    # Ensure both keys exist and normalize
    ratios = {k: float(ratios.get(k, 0.0)) for k in ["K", "P"]}
    total_r = ratios["K"] + ratios["P"]
    if total_r <= 0:
        ratios = {"K": 0.7, "P": 0.3}
    else:
        ratios = {k: ratios[k] / total_r for k in ratios}

    alloc = compute_target_allocation(top_k, ratios)
    logger.info(f"Balancing allocation: K={alloc.get('K', 0)}, P={alloc.get('P', 0)}, O={alloc.get('O', 0)} for top_k={top_k}")
    logger.info(f"Available candidates: K={len(cat_lists['K'])}, P={len(cat_lists['P'])}, O={len(cat_lists['O'])}")

    result = []
    
    # Greedy pick from each bucket up to allocation
    for typ in ["K", "P", "O"]:
        take = alloc.get(typ, 0)
        available = cat_lists.get(typ, [])
        take = min(take, len(available))
        result.extend(available[:take])
        logger.debug(f"Picked {take} from type {typ}")

    # If we don't have enough (e.g., not enough P), append highest remaining
    if len(result) < top_k:
        remaining = []
        for t in cat_lists:
            remaining.extend(cat_lists[t][alloc.get(t, 0):])
        remaining.sort(key=lambda x: -float(x.get("score", x.get("hybrid_score", 0.0))))
        need = top_k - len(result)
        result.extend(remaining[:need])
        logger.debug(f"Filled {need} slots from remaining candidates")

    # Final sort by score to keep highest relevance within chosen mix
    result.sort(key=lambda x: -float(x.get("score", x.get("hybrid_score", 0.0))))
    
    logger.info(f"Final balanced result: {len(result)} candidates")
    
    return result[:top_k]
