"""
Critical Improvements to Address System Weaknesses

This module implements fixes for 5 key issues:
1. Finance Analyst recommendations (Customer Service) - Role mismatch
2. Duplicate entries (same URL repeated) - Unprofessional output
3. No duration filtering - Violates query constraints
4. Generic solutions over-recommended - Poor relevance
5. Presales role mismatch - Missing specialized assessments

Implementation Strategy:
- Add role-specific domain mappings for analytical/financial/specialist roles
- Implement URL deduplication in final results
- Extract and apply duration constraints from queries
- Increase specificity weight and add generic penalty
- Add specialized role templates for better matching
"""

import re
from typing import List, Dict, Optional, Set
from src.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# FIX 1: Role-Specific Domain Mappings (Finance Analyst Issue)
# ============================================================================

SPECIALIST_ROLE_DOMAINS = {
    # Financial & Analytical Roles
    "finance": ["financial", "accounting", "budgeting", "forecasting", "excel", "quickbooks", "sap", "oracle"],
    "analyst": ["analysis", "data", "research", "insights", "statistics", "reporting", "analytics", "business intelligence"],
    "financial analyst": ["financial modeling", "valuation", "investment", "portfolio", "risk management", "capital markets"],
    "data analyst": ["sql", "python", "tableau", "power bi", "data visualization", "statistics", "excel"],
    "business analyst": ["requirements", "stakeholder", "process", "uml", "agile", "user stories", "documentation"],
    
    # Sales & Presales Roles
    "presales": ["solution", "technical sales", "demo", "proof of concept", "rfp", "proposal", "product knowledge"],
    "sales engineer": ["technical", "solution", "demo", "customer", "integration", "architecture"],
    "account manager": ["relationship", "retention", "upsell", "customer success", "negotiation"],
    
    # Specialized Technical Roles
    "devops": ["ci/cd", "kubernetes", "docker", "jenkins", "terraform", "cloud", "automation"],
    "security": ["cybersecurity", "penetration", "vulnerability", "compliance", "encryption", "threat"],
    "qa": ["testing", "automation", "selenium", "test cases", "quality", "bug", "regression"],
    
    # Content & Marketing
    "content writer": ["seo", "copywriting", "cms", "wordpress", "content strategy", "editorial"],
    "marketing": ["campaigns", "social media", "email", "analytics", "branding", "advertising"],
    
    # HR & Operations
    "hr": ["recruitment", "talent", "onboarding", "performance", "compensation", "employee relations"],
    "operations": ["process", "efficiency", "logistics", "supply chain", "inventory", "workflow"],
}


def get_domain_keywords(role: str, skills: List[str]) -> List[str]:
    """
    Get domain-specific keywords for a role to improve matching.
    
    Args:
        role: Detected role from query (e.g., "Finance Analyst")
        skills: List of skills mentioned in query
        
    Returns:
        List of domain keywords to boost in search
    """
    role_lower = role.lower()
    keywords = []
    
    # Find matching domain
    for domain, domain_keywords in SPECIALIST_ROLE_DOMAINS.items():
        if domain in role_lower or role_lower in domain:
            keywords.extend(domain_keywords)
            logger.info(f"Matched domain '{domain}' for role '{role}' → Adding {len(domain_keywords)} keywords")
            break
    
    # Add skills as keywords
    keywords.extend([skill.lower() for skill in skills])
    
    return list(set(keywords))  # Remove duplicates


# ============================================================================
# FIX 2: URL Deduplication (Duplicate Entries Issue)
# ============================================================================

def deduplicate_by_url(results: List[Dict]) -> List[Dict]:
    """
    Remove duplicate assessments by URL, keeping the highest-ranked one.
    
    Args:
        results: List of assessment dicts with 'url' field
        
    Returns:
        Deduplicated list maintaining original ranking order
    """
    seen_urls: Set[str] = set()
    deduped = []
    duplicates_removed = 0
    
    for result in results:
        url = result.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(result)
        elif url in seen_urls:
            duplicates_removed += 1
            logger.warning(f"Removed duplicate URL: {url} ({result.get('name', 'Unknown')})")
    
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate entries from {len(results)} results")
    
    return deduped


# ============================================================================
# FIX 3: Duration Constraint Extraction (Duration Filtering Issue)
# ============================================================================

DURATION_PATTERNS = [
    # "under X minutes", "less than X minutes", "max X minutes"
    (r'(?:under|less than|max|maximum|up to)\s+(\d+)\s*(?:min|minutes)', 'max'),
    
    # "at least X minutes", "minimum X minutes", "longer than X"
    (r'(?:at least|minimum|min|longer than)\s+(\d+)\s*(?:min|minutes)', 'min'),
    
    # "X-Y minutes", "between X and Y minutes"
    (r'(?:between\s+)?(\d+)\s*(?:-|to|and)\s*(\d+)\s*(?:min|minutes)', 'range'),
    
    # "X minute", "X minutes" (exact)
    (r'(?:exactly\s+)?(\d+)\s*(?:min|minutes)\s+(?:long|duration|assessment)', 'exact'),
]


def extract_duration_constraint(query: str) -> Optional[Dict[str, int]]:
    """
    Extract duration preferences from query text (for soft scoring, not filtering).
    
    Args:
        query: User query text
        
    Returns:
        Dict with 'min' and/or 'max' keys, or None if no constraint found
        Example: {"max": 30} or {"min": 15, "max": 45}
    """
    query_lower = query.lower()
    constraint = {}
    
    for pattern, constraint_type in DURATION_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            if constraint_type == 'max':
                constraint['max'] = int(match.group(1))
                logger.info(f"Extracted max duration preference: {constraint['max']} minutes")
                
            elif constraint_type == 'min':
                constraint['min'] = int(match.group(1))
                logger.info(f"Extracted min duration preference: {constraint['min']} minutes")
                
            elif constraint_type == 'range':
                constraint['min'] = int(match.group(1))
                constraint['max'] = int(match.group(2))
                logger.info(f"Extracted duration range preference: {constraint['min']}-{constraint['max']} minutes")
                
            elif constraint_type == 'exact':
                exact = int(match.group(1))
                # Much wider tolerance - ±20 minutes
                constraint['min'] = max(exact - 20, 0)
                constraint['max'] = exact + 20
                logger.info(f"Extracted duration preference: ~{exact} minutes (tolerance ±20)")
            
            break  # Use first match
    
    return constraint if constraint else None


def calculate_duration_score(assessment_duration: int, constraint: Dict[str, int]) -> float:
    """
    Calculate soft score [0.3, 1.0] based on duration match.
    
    Perfect match = 1.0, Far away = 0.3 (but NOT filtered out)
    
    Args:
        assessment_duration: Duration in minutes
        constraint: Dict with 'min' and/or 'max' duration preferences
        
    Returns:
        Score between 0.3 and 1.0
    """
    if not constraint:
        return 1.0  # No constraint = perfect score
    
    # Calculate target duration (midpoint of range)
    min_dur = constraint.get('min', 0)
    max_dur = constraint.get('max', float('inf'))
    
    if max_dur == float('inf'):
        target = min_dur
    else:
        target = (min_dur + max_dur) / 2
    
    # Calculate difference from target
    difference = abs(assessment_duration - target)
    
    # Soft decay: Within 15 min = high score, beyond 40 min = low (but still included)
    if difference <= 15:
        return 1.0  # Perfect
    elif difference <= 25:
        return 0.85  # Very good
    elif difference <= 35:
        return 0.7  # Good
    elif difference <= 50:
        return 0.5  # Acceptable
    else:
        return 0.3  # Far but still included


def apply_duration_filter(results: List[Dict], constraint: Dict[str, int]) -> List[Dict]:
    """
    Apply SOFT duration scoring (NO HARD FILTERING).
    
    This function now only reorders results by duration preference,
    it does NOT remove any results.
    
    Args:
        results: List of assessment dicts with 'duration' or 'duration_minutes' field
        constraint: Dict with 'min' and/or 'max' duration preferences
        
    Returns:
        Results with duration scores applied and reordered (ALL results kept)
    """
    if not constraint:
        logger.info("No duration constraint - returning all results")
        return results
    
    scored_results = []
    for result in results:
        # Try both field names
        duration = result.get('duration_minutes') or result.get('duration')
        
        # Handle various duration formats
        if isinstance(duration, str):
            # Extract numeric value from strings like "30 minutes" or "30-45"
            match = re.search(r'(\d+)', duration)
            if match:
                duration = int(match.group(1))
            else:
                logger.debug(f"Could not parse duration: {duration} for {result.get('name')}")
                duration = None
        
        if isinstance(duration, (int, float)):
            # Calculate duration score
            duration_score = calculate_duration_score(int(duration), constraint)
            
            # Apply duration score to existing score (5% weight - very light)
            original_score = result.get('score', result.get('final_score', 1.0))
            
            # Weight: 95% original score + 5% duration score (very light preference)
            result['score'] = original_score * 0.95 + duration_score * 0.05
            result['duration_score'] = duration_score
            
            logger.debug(f"{result.get('name')}: duration={duration}min, "
                        f"duration_score={duration_score:.2f}, "
                        f"final_score={result['score']:.2f}")
        else:
            # No duration info - keep original score
            result['duration_score'] = 1.0
        
        scored_results.append(result)
    
    # Re-sort by updated scores
    scored_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    logger.info(f"Duration soft scoring applied: {len(results)} results kept (constraint: {constraint})")
    return scored_results


# ============================================================================
# FIX 4: Generic Assessment Penalty (Generic Over-Recommendation Issue)
# ============================================================================

GENERIC_INDICATORS = [
    # Generic test names (always penalize these)
    "general", "basic", "fundamental", "introductory", "overview", "essentials",
    
    # Broad category names without specifics
    "software", "technology", "computer", "digital", "online",
    
    # Very common skills that don't indicate specialization
    "communication", "teamwork", "problem solving", "critical thinking",
]

SPECIALIZED_INDICATORS = [
    # Specific technologies/tools
    "python", "java", "javascript", "sql", "react", "angular", "aws", "azure",
    "docker", "kubernetes", "terraform", "jenkins", "git",
    
    # Domain-specific terms
    "financial modeling", "accounting", "budget", "forecasting",
    "devops", "cybersecurity", "machine learning", "data science",
    "salesforce", "sap", "oracle", "quickbooks",
    
    # Advanced concepts
    "advanced", "professional", "expert", "senior", "architect",
    "optimization", "algorithm", "design pattern", "architecture",
]


def calculate_specificity_penalty(assessment_name: str, assessment_desc: str, query: str) -> float:
    """
    Calculate penalty for generic assessments that don't match specialized queries.
    
    Returns:
        Penalty multiplier between 0.5 (generic) and 1.0 (specialized)
    """
    text = f"{assessment_name} {assessment_desc}".lower()
    query_lower = query.lower()
    
    # Count generic vs specialized indicators
    generic_count = sum(1 for indicator in GENERIC_INDICATORS if indicator in text)
    specialized_count = sum(1 for indicator in SPECIALIZED_INDICATORS if indicator in text)
    
    # Check if query is specialized (contains domain-specific terms)
    query_is_specialized = any(indicator in query_lower for indicator in SPECIALIZED_INDICATORS)
    
    # Apply penalty if assessment is generic but query is specialized
    if query_is_specialized and generic_count > specialized_count:
        penalty = 0.6  # 40% penalty for generic assessments
        logger.debug(f"Generic penalty: {assessment_name} → {penalty}x (generic={generic_count}, specialized={specialized_count})")
        return penalty
    
    # Slight penalty for any generic assessment
    if generic_count > 0:
        return 0.9
    
    return 1.0  # No penalty


def rerank_with_specificity_penalty(results: List[Dict], query: str) -> List[Dict]:
    """
    Rerank results by applying specificity penalty to generic assessments.
    
    Args:
        results: List of assessment dicts with scores
        query: Original query text
        
    Returns:
        Reranked list with updated scores
    """
    for result in results:
        name = result.get('name', '')
        desc = result.get('description', '')
        penalty = calculate_specificity_penalty(name, desc, query)
        
        # Apply penalty to score (if available)
        if 'score' in result:
            original_score = result['score']
            result['score'] = original_score * penalty
            if penalty < 1.0:
                logger.debug(f"Penalized {name}: {original_score:.3f} → {result['score']:.3f}")
    
    # Re-sort by updated scores
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    return results


# ============================================================================
# FIX 5: Role Template Matching (Presales/Specialist Role Issue)
# ============================================================================

ROLE_ASSESSMENT_TEMPLATES = {
    # Map roles to required assessment characteristics
    "presales": {
        "required_keywords": ["solution", "technical", "product", "demo", "sales"],
        "required_test_types": ["Knowledge", "Performance"],
        "preferred_assessments": ["solution architect", "technical sales", "product knowledge"],
    },
    
    "finance analyst": {
        "required_keywords": ["financial", "accounting", "analysis", "excel", "modeling"],
        "required_test_types": ["Knowledge", "Performance"],
        "preferred_assessments": ["financial", "accounting", "excel", "analysis"],
    },
    
    "data analyst": {
        "required_keywords": ["sql", "data", "analysis", "statistics", "visualization"],
        "required_test_types": ["Knowledge", "Performance"],
        "preferred_assessments": ["sql", "python", "data", "tableau", "power bi"],
    },
    
    "security": {
        "required_keywords": ["security", "cybersecurity", "penetration", "vulnerability"],
        "required_test_types": ["Knowledge"],
        "preferred_assessments": ["security", "cybersecurity", "ethical hacking"],
    },
}


def match_role_template(results: List[Dict], detected_role: str, query: str) -> List[Dict]:
    """
    Boost results that match role-specific templates.
    
    Args:
        results: List of assessment dicts
        detected_role: Role detected from query
        query: Original query text
        
    Returns:
        Reranked results with role-specific boosting
    """
    role_lower = detected_role.lower()
    
    # Find matching template
    template = None
    for role_key, role_template in ROLE_ASSESSMENT_TEMPLATES.items():
        if role_key in role_lower or role_lower in role_key:
            template = role_template
            logger.info(f"Matched role template for '{detected_role}' → {role_key}")
            break
    
    if not template:
        return results  # No template found, return as-is
    
    # Score each result by template match
    for result in results:
        name_lower = result.get('name', '').lower()
        desc_lower = result.get('description', '').lower()
        text = f"{name_lower} {desc_lower}"
        
        boost_score = 1.0
        
        # Check required keywords
        keyword_matches = sum(1 for kw in template['required_keywords'] if kw in text)
        if keyword_matches > 0:
            boost_score += 0.2 * keyword_matches
        
        # Check preferred assessments
        pref_matches = sum(1 for pref in template['preferred_assessments'] if pref in name_lower)
        if pref_matches > 0:
            boost_score += 0.3 * pref_matches
        
        # Apply boost
        if boost_score > 1.0:
            original_score = result.get('score', 0)
            result['score'] = original_score * boost_score
            logger.debug(f"Role template boost: {result.get('name')} → {original_score:.3f} × {boost_score:.2f} = {result['score']:.3f}")
    
    # Re-sort by updated scores
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    return results


# ============================================================================
# Master Improvement Pipeline
# ============================================================================

def apply_improvements(
    results: List[Dict],
    query: str,
    detected_role: Optional[str] = None,
    top_k: int = 10
) -> List[Dict]:
    """
    Apply all 5 critical improvements to recommendation results.
    
    Pipeline:
    1. Remove duplicate URLs
    2. Apply duration filtering (if constraint in query)
    3. Apply specificity penalty to generic assessments
    4. Apply role template matching boost
    5. Take top-k after all adjustments
    
    Args:
        results: Raw results from retrieval/reranking
        query: Original user query
        detected_role: Role detected from query (optional)
        top_k: Number of final results to return
        
    Returns:
        Improved and filtered results list
    """
    logger.info(f"Applying improvements pipeline to {len(results)} results")
    
    # Fix 2: Remove duplicates
    results = deduplicate_by_url(results)
    
    # Fix 3: Apply duration filtering
    duration_constraint = extract_duration_constraint(query)
    if duration_constraint:
        results = apply_duration_filter(results, duration_constraint)
    
    # Fix 4: Penalize generic assessments
    results = rerank_with_specificity_penalty(results, query)
    
    # Fix 5: Apply role template matching
    if detected_role:
        results = match_role_template(results, detected_role, query)
    
    # Return top-k after improvements
    final_results = results[:top_k]
    logger.info(f"Improvements complete: {len(results)} → {len(final_results)} results returned")
    
    return final_results


# ============================================================================
# NEW: Domain Prefiltering, Specialist Validation, and Combined Scoring
# ============================================================================

FINANCE_ANALYTICS_DOMAINS = {
    "finance",
    "financial",
    "accounting",
    "analyst",
    "analytics",
    "data",
    "quant",
    "reporting",
    "excel",
    "sql"
}

SPECIALIST_ROLE_KEYWORDS = {
    "presales": ["presales", "pre-sales", "sales engineer", "solutions consultant"],
    "research": ["research", "researcher"],
    "scientist": ["scientist"],
    "consultant": ["consultant"],
}


def extract_domains_from_query(query_info: Optional[Dict]) -> List[str]:
    """
    Extract coarse domains/role intents from query_info.
    """
    if not query_info:
        return []
    role = (query_info.get("role") or "").lower()
    skills = [s.lower() for s in (query_info.get("skills") or [])]
    tokens = set()
    for tok in role.split():
        tokens.add(tok)
    for s in skills:
        tokens.update(s.split())
    # Focus on finance/analytics tokens (extensible)
    domains = [d for d in FINANCE_ANALYTICS_DOMAINS if d in tokens or any(d in s for s in skills) or d in role]
    return list(dict.fromkeys(domains))  # preserve order, unique


def prefilter_candidates_by_domain(candidates: List[Dict], required_domains: List[str]) -> List[Dict]:
    """
    Keep only candidates that mention at least one required domain
    in description or test_types (case-insensitive). Fallback if too few.
    """
    if not candidates or not required_domains:
        return candidates
    req = [d.lower() for d in required_domains]
    filtered = []
    for c in candidates:
        desc = (c.get("description", "") or "").lower()
        name = (c.get("name", "") or "").lower()
        tt = c.get("test_types", [])
        if isinstance(tt, str):
            tt_text = tt.lower()
        else:
            tt_text = " ".join(map(lambda x: str(x).lower(), tt or []))
        blob = f"{name} {desc} {tt_text}"
        if any(d in blob for d in req):
            filtered.append(c)
    # Only apply if we still have a healthy pool
    return filtered if len(filtered) >= 10 else candidates


def extract_specialist_keywords(query_info: Optional[Dict]) -> List[str]:
    """
    Extract specialist keywords from detected role.
    """
    if not query_info:
        return []
    role = (query_info.get("role") or "").lower()
    kws: List[str] = []
    for key, vals in SPECIALIST_ROLE_KEYWORDS.items():
        if key in role:
            kws.extend(vals)
    # Also include the role token itself
    role_tokens = [t for t in role.split() if len(t) > 2]
    kws.extend(role_tokens)
    # De-duplicate while preserving order
    seen = set()
    out = []
    for k in kws:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def post_rerank_specialist_validate(reranked: List[Dict], specialist_keywords: List[str], top_k: int) -> List[Dict]:
    """
    Ensure top results contain specialist keywords; fill remainder from reranked.
    """
    if not reranked or not specialist_keywords:
        return reranked[:top_k]
    kws = [k.lower() for k in specialist_keywords]
    final_results: List[Dict] = []
    for r in reranked:
        text = f"{r.get('name','')} {r.get('description','')}".lower()
        if any(k in text for k in kws):
            final_results.append(r)
            if len(final_results) >= top_k:
                break
    if len(final_results) < top_k:
        for r in reranked:
            if r not in final_results:
                final_results.append(r)
                if len(final_results) >= top_k:
                    break
    return final_results[:top_k]


def domain_match(result: Dict, query_info: Optional[Dict]) -> bool:
    """
    Check if a result matches any extracted domains from the query.
    """
    domains = extract_domains_from_query(query_info)
    if not domains:
        return False
    text = f"{result.get('name','')} {result.get('description','')}"
    tt = result.get("test_types", [])
    if isinstance(tt, str):
        text = f"{text} {tt}"
    else:
        text = f"{text} {' '.join(map(str, tt or []))}"
    text = text.lower()
    return any(d in text for d in domains)


def combine_scores_with_domain(reranked: List[Dict], query_info: Optional[Dict], top_k: int = 10) -> List[Dict]:
    """
    Combine LLM rerank order with hybrid retrieval score and domain bonus.
    If llm_score not provided, infer it from rank position.
    """
    if not reranked:
        return []
    n = len(reranked)
    for idx, r in enumerate(reranked):
        llm_score = r.get("llm_score")
        if llm_score is None:
            # Infer: top gets 1.0, last gets ~0.0
            llm_score = 1.0 - (idx / max(1, n - 1)) if n > 1 else 1.0
        hybrid_score = float(r.get("final_score", r.get("score", 0.0)))
        bonus = 0.1 if domain_match(r, query_info) else 0.0
        r["combined_score"] = 0.5 * float(llm_score) + 0.4 * hybrid_score + bonus
    reranked.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return reranked[:top_k]