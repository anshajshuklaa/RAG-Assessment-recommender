"""
LangGraph-based Workflow Orchestration with State Machine

Implements explicit extract→rag→filter graph with:
- Confidence checks at each stage
- Retry logic for low-confidence results
- Fallback to broader retrieval strategies
- State tracking and branching decisions
"""

from typing import TypedDict, List, Dict, Literal, Optional
from langgraph.graph import StateGraph, END
import asyncio
import pandas as pd
import json
from pathlib import Path

from src.query_enhancer import HybridQueryEnhancer
from src.retriever import HybridRetriever
from src.gemini_reranker import GeminiReranker
from src.test_type_balancer import balance_candidates, default_ratio_from_query
from src.improvements import apply_improvements
from src.improvements import (
    extract_domains_from_query,
    prefilter_candidates_by_domain,
    extract_specialist_keywords,
    combine_scores_with_domain,
    post_rerank_specialist_validate,
)
from src.logging_config import get_logger
from src.degradation import start_request

logger = get_logger(__name__)

# Load assessment metadata once at module level
_assessment_metadata = None

def load_assessment_metadata() -> Dict:
    """Load assessment metadata from JSON file and convert to dict by name."""
    global _assessment_metadata
    if _assessment_metadata is None:
        metadata_path = Path(__file__).parent / "data" / "assessments_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Assessment metadata not found at {metadata_path}. "
                "This file is required for the application to run. "
                "Please ensure data/assessments_metadata.json is present."
            )
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        
        # Convert list to dict with name as key (for matching with results)
        _assessment_metadata = {}
        for idx, item in enumerate(metadata_list):
            # Use both index and name as keys for flexible matching
            _assessment_metadata[idx] = item
            _assessment_metadata[item.get("name", f"unknown_{idx}")] = item
        
        logger.info(f"Loaded {len(metadata_list)} assessment metadata entries")
    return _assessment_metadata


class WorkflowState(TypedDict):
    """
    State object passed through the workflow graph.
    
    Attributes:
        query: Original user query
        enhanced_query: Query after enhancement
        query_info: Metadata from query enhancement (role, skills, etc.)
        retrieval_results: List of retrieved candidates
        retrieval_confidence: Confidence score for retrieval quality (0-1)
        reranked_results: Final reranked results
        reranker_confidence: Confidence score for reranking quality (0-1)
        retry_count: Number of retry attempts made
        strategy: Current retrieval strategy (focused/broad)
        error: Error message if any step failed
        custom_test_type_ratio: Optional custom K/P ratio from API request
    """
    query: str
    enhanced_query: Optional[str]
    query_info: Optional[Dict]
    retrieval_results: Optional[List[Dict]]
    retrieval_confidence: float
    reranked_results: Optional[List[Dict]]
    reranker_confidence: float
    retry_count: int
    strategy: Literal["focused", "broad"]
    error: Optional[str]
    custom_test_type_ratio: Optional[Dict[str, float]]


class WorkflowOrchestrator:
    """
    LangGraph-based workflow orchestrator with confidence checks and retries.
    
    Workflow stages:
    1. Extract: Query enhancement and information extraction
    2. RAG: Hybrid retrieval with confidence scoring
    3. Filter: Intelligent reranking with confidence checks
    4. Retry logic: Fallback to broader retrieval if confidence is low
    """
    
    # Confidence thresholds
    MIN_RETRIEVAL_CONFIDENCE = 0.5  # Minimum confidence for retrieval results
    MIN_RERANKER_CONFIDENCE = 0.7   # Minimum confidence for reranked results
    MAX_RETRY_COUNT = 2              # Maximum retry attempts
    
    def __init__(self):
        """Initialize workflow components."""
        logger.info("Initializing LangGraph Workflow Orchestrator")
        
        self.query_enhancer = HybridQueryEnhancer()
        self.retriever = HybridRetriever()
        self.reranker = GeminiReranker()
        
        # Build the workflow graph
        self.graph = self._build_graph()
        logger.info("Workflow graph initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine.
        
        Graph structure:
            START → extract → retrieval_check → rag → reranking_check → filter → END
                                    ↓                           ↓
                              retry_with_broad          retry_with_rerank
                                    ↓                           ↓
                                  rag ←─────────────────────────┘
        """
        workflow = StateGraph(WorkflowState)
        
        # Define nodes (stages)
        workflow.add_node("extract", self._extract_node)
        workflow.add_node("rag", self._rag_node)
        workflow.add_node("filter", self._filter_node)
        workflow.add_node("retry_with_broad", self._retry_broad_node)
        
        # Define edges and conditional routing
        workflow.set_entry_point("extract")
        
        # After extraction, always go to RAG
        workflow.add_edge("extract", "rag")
        
        # After RAG, check confidence and decide
        workflow.add_conditional_edges(
            "rag",
            self._should_retry_retrieval,
            {
                "retry": "retry_with_broad",
                "continue": "filter"
            }
        )
        
        # After retry, go back to RAG with new strategy
        workflow.add_edge("retry_with_broad", "rag")
        
        # After filtering, check confidence and decide
        workflow.add_conditional_edges(
            "filter",
            self._should_retry_reranking,
            {
                "retry": "retry_with_broad",
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def _extract_node(self, state: WorkflowState) -> WorkflowState:
        """
        Stage 1: Query Enhancement and Information Extraction.
        
        Extracts role, skills, and preferences from the query.
        """
        logger.info("Stage 1: Extract - Enhancing query")
        
        try:
            result = self.query_enhancer.enhance(state["query"])
            
            state["enhanced_query"] = result["enhanced"]
            state["query_info"] = {
                "role": result.get("role", ""),
                "skills": result.get("all_skills", []),
                "seniority": result.get("seniority", ""),
                "preferences": result.get("preferences", []),
                "test_types": result.get("test_types", [])
            }
            
            logger.info(f"  ✓ Query enhanced successfully (role={result.get('role', 'N/A')}, "
                       f"skills={len(result.get('all_skills', []))})")
            
        except Exception as e:
            logger.error(f"  ✗ Query enhancement failed: {e}", exc_info=True)
            state["error"] = f"Enhancement failed: {str(e)}"
            state["enhanced_query"] = state["query"]  # Fallback to original
            state["query_info"] = {}
        
        return state
    
    async def _rag_node(self, state: WorkflowState) -> WorkflowState:
        """
        Stage 2: Hybrid RAG Retrieval.
        
        Retrieves candidates using semantic + BM25 + specificity scoring.
        Calculates confidence based on top scores and diversity.
        """
        strategy = state.get("strategy", "focused")
        logger.info(f"Stage 2: RAG - Retrieving with {strategy} strategy")
        
        try:
            # Adjust retrieval parameters based on strategy
            if strategy == "broad":
                # Broader retrieval: more candidates, lower thresholds
                k = 100
                logger.info("  Using broad retrieval (100 candidates)")
            else:
                # Focused retrieval: standard parameters
                k = 50
                logger.info("  Using focused retrieval (50 candidates)")
            
            # Perform retrieval (returns (index, score_dict) tuples)
            candidates = await asyncio.to_thread(
                self.retriever.retrieve,
                query=state["enhanced_query"] or state["query"],
                k=k,
                return_scores=True
            )
            
            # Scale hybrid scores to [0, 1] so downstream weighting and confidence are comparable
            max_score = max((sc["hybrid"] for _, sc in candidates), default=0.0) or 1.0
            
            # Convert indices to full assessment data
            results = []
            for idx, score_dict in candidates:
                row = self.retriever.df.iloc[idx]
                duration = row.get('duration_minutes')
                results.append({
                    "name": row['name'],
                    "url": row.get('url', ''),
                    "description": row.get('description', ''),
                    "duration": int(duration) if pd.notna(duration) else 0,
                    "test_types": row.get('test_types', ''),
                    "adaptive_irt": str(row.get('adaptive_irt_support', '')).strip().lower() == 'yes',
                    "remote_testing": str(row.get('remote_testing_support', '')).strip().lower() == 'yes',
                    "final_score": score_dict["hybrid"] / max_score,
                    "raw_score": score_dict["hybrid"]
                })
            
            state["retrieval_results"] = results
            
            # Calculate retrieval confidence
            confidence = self._calculate_retrieval_confidence(results)
            state["retrieval_confidence"] = confidence
            
            logger.info(f"  ✓ Retrieved {len(results)} candidates (confidence={confidence:.2f})")
            
        except Exception as e:
            logger.error(f"  ✗ Retrieval failed: {e}", exc_info=True)
            state["error"] = f"Retrieval failed: {str(e)}"
            state["retrieval_results"] = []
            state["retrieval_confidence"] = 0.0
        
        return state
    
    async def _filter_node(self, state: WorkflowState) -> WorkflowState:
        """
        Stage 3: Intelligent Reranking Filter with Test Type Balancing.
        
        1. Detects collaboration/soft-skill intent in query
        2. Applies test type balancing (K/P ratio) before reranking
        3. Intelligently reranks balanced candidates
        4. Calculates confidence based on score distribution and top results
        """
        logger.info("Stage 3: Filter - Test Type Balancing + intelligent reranking")
        
        try:
            results = state.get("retrieval_results", [])
            
            if not results:
                logger.warning("  ⚠ No results to rerank")
                state["reranked_results"] = []
                state["reranker_confidence"] = 0.0
                return state
            
            # Load metadata for test type balancing
            metadata = load_assessment_metadata()
            
            # Convert results to format expected by balancer (with ID and score)
            # Match results with metadata by name (metadata dict has names as keys)
            candidates_with_ids = []
            for idx, r in enumerate(results):
                name = r.get("name", "")
                matched_meta = metadata.get(name)
                
                if matched_meta:
                    candidates_with_ids.append({
                        "id": name,  # Use name as ID for matching
                        "name": name,
                        "score": r.get("final_score", r.get("score", 0.0)),
                        "url": r.get("url", ""),
                        "description": r.get("description", ""),
                        "duration": r.get("duration", 0),
                        "test_types": matched_meta.get("test_types", [])  # Get from metadata
                    })
                else:
                    # If no metadata match, still include with original data
                    candidates_with_ids.append({
                        "id": name,
                        "name": name,
                        "score": r.get("final_score", r.get("score", 0.0)),
                        "url": r.get("url", ""),
                        "description": r.get("description", ""),
                        "duration": r.get("duration", 0),
                        "test_types": r.get("test_types", "")
                    })
            
            logger.info(f"  → Prepared {len(candidates_with_ids)} candidates for balancing")
            
            # Detect collaboration intent and determine target ratio
            query_info = state.get("query_info") or {}

            # Domain pre-filtering for analytical/financial roles (before balancing/reranking)
            try:
                required_domains = extract_domains_from_query(query_info)
                if required_domains:
                    pre_len = len(candidates_with_ids)
                    filtered_candidates = prefilter_candidates_by_domain(candidates_with_ids, required_domains)
                    if len(filtered_candidates) >= 10:
                        candidates_with_ids = filtered_candidates
                        logger.info(f"  → Domain pre-filter applied ({pre_len} → {len(candidates_with_ids)}) for domains={required_domains}")
                    else:
                        logger.info("  → Domain pre-filter skipped due to insufficient candidates")
            except Exception as e:
                logger.warning(f"  → Domain pre-filter failed ({e}); proceeding without it")
            
            # Log query info for debugging
            logger.info(f"  → Query info: role={query_info.get('role', 'N/A')}, "
                       f"skills={query_info.get('skills', [])[:5]}")
            
            # Use custom ratio from API if provided, otherwise auto-detect
            custom_ratio = state.get("custom_test_type_ratio")
            if custom_ratio:
                logger.info(f"  → Using API-provided test type ratio: K={custom_ratio.get('K', 0):.0%}, "
                           f"P={custom_ratio.get('P', 0):.0%}")
            else:
                custom_ratio = default_ratio_from_query(query_info)
                logger.info(f"  → Auto-detected test type ratio: K={custom_ratio.get('K', 0):.0%}, "
                           f"P={custom_ratio.get('P', 0):.0%}")
            
            # Apply test type balancing to create a balanced shortlist
            # The metadata dict is keyed by name, so the balancer can look up by name directly
            balanced_candidates = balance_candidates(
                candidates=candidates_with_ids[:50],  # Use top 50 from retrieval
                assessment_metadata=metadata,
                enhanced_query=query_info,
                top_k=30,  # Create balanced shortlist of 30 for reranking
                custom_ratio=custom_ratio
            )
            
            # Log the distribution after balancing
            from collections import Counter
            dist = Counter()
            for c in balanced_candidates:
                meta = metadata.get(c.get("id"), {})
                test_types = meta.get("test_types", [])
                if isinstance(test_types, str):
                    test_types = [test_types]
                for tt in test_types:
                    dist[tt.lower() if isinstance(tt, str) else str(tt)] += 1
            
            logger.info(f"  → Balanced candidate distribution: {dict(dist)}")
            
            # Convert back to format expected by reranker
            reranker_input = []
            for c in balanced_candidates:
                reranker_input.append({
                    "name": c.get("name", ""),
                    "url": c.get("url", ""),
                    "description": c.get("description", ""),
                    "duration": c.get("duration", 0),
                    "test_types": c.get("test_types", ""),
                    "final_score": c.get("score", 0.0)
                })
            
            # Rerank the balanced shortlist
            logger.info(f"  → Reranking {len(reranker_input)} balanced candidates")
            reranked = await asyncio.to_thread(
                self.reranker.rerank,
                state["query"],
                reranker_input,
                top_k=15  # Get top-15 before applying improvements
            )

            # Combine reranked order with hybrid score and domain bonus
            try:
                combined_ranked = combine_scores_with_domain(reranked, query_info, top_k=15)
            except Exception as e:
                logger.warning(f"  → Combined scoring failed ({e}); using reranked results")
                combined_ranked = reranked

            # Post-reranking validation for specialist roles
            try:
                specialist_keywords = extract_specialist_keywords(query_info)
                if specialist_keywords:
                    validated = post_rerank_specialist_validate(combined_ranked, specialist_keywords, top_k=15)
                else:
                    validated = combined_ranked[:15]
            except Exception as e:
                logger.warning(f"  → Specialist validation failed ({e}); proceeding without it")
                validated = combined_ranked[:15]
            
            # Apply critical improvements pipeline
            logger.info("  → Applying improvements pipeline (dedup, duration filter, specificity, role match)")
            improved_results = apply_improvements(
                results=validated,
                query=state["query"],
                detected_role=query_info.get("role"),
                top_k=10  # Final top-10 after improvements
            )
            
            # Re-attach catalogue fields that the balancing/reranking steps drop
            by_name = {r["name"]: r for r in results}
            for r in improved_results:
                source = by_name.get(r.get("name"), {})
                r.setdefault("adaptive_irt", source.get("adaptive_irt", False))
                r.setdefault("remote_testing", source.get("remote_testing", False))
            
            state["reranked_results"] = improved_results
            
            # Calculate reranking confidence
            confidence = self._calculate_reranking_confidence(validated, query_info)
            state["reranker_confidence"] = confidence
            
            # Log final distribution
            final_dist = Counter()
            for r in validated:
                test_types = r.get("test_types", "")
                if isinstance(test_types, list):
                    for tt in test_types:
                        final_dist[tt.lower() if isinstance(tt, str) else str(tt)] += 1
                else:
                    final_dist[str(test_types).lower()] += 1
            
            logger.info(f"  ✓ Reranked to top-{len(reranked)} (confidence={confidence:.2f})")
            logger.info(f"  ✓ Final distribution: {dict(final_dist)}")
            
        except Exception as e:
            logger.error(f"  ✗ Reranking failed: {e}", exc_info=True)
            state["error"] = f"Reranking failed: {str(e)}"
            # Fallback: use retrieval results
            retrieval_results = state.get("retrieval_results") or []
            state["reranked_results"] = retrieval_results[:10]
            state["reranker_confidence"] = 0.5
        
        return state
    
    async def _retry_broad_node(self, state: WorkflowState) -> WorkflowState:
        """
        Retry node: Switch to broader retrieval strategy.
        
        Increments retry count and changes strategy to 'broad' for next RAG attempt.
        """
        state["retry_count"] = state.get("retry_count", 0) + 1
        state["strategy"] = "broad"
        
        logger.warning(f"  ⚠ Low confidence detected, switching to broad retrieval "
                      f"(retry {state['retry_count']}/{self.MAX_RETRY_COUNT})")
        
        return state
    
    def _should_retry_retrieval(self, state: WorkflowState) -> Literal["retry", "continue"]:
        """
        Decision: Should we retry retrieval with broader strategy?
        
        Retry if:
        - Retrieval confidence is below threshold
        - Haven't exceeded max retries
        - Using focused strategy (can broaden)
        """
        confidence = state.get("retrieval_confidence", 0.0)
        retry_count = state.get("retry_count", 0)
        strategy = state.get("strategy", "focused")
        
        if (confidence < self.MIN_RETRIEVAL_CONFIDENCE and 
            retry_count < self.MAX_RETRY_COUNT and 
            strategy == "focused"):
            logger.warning(f"  → Retrieval confidence {confidence:.2f} < {self.MIN_RETRIEVAL_CONFIDENCE}, will retry")
            return "retry"
        
        logger.info(f"  → Retrieval confidence {confidence:.2f} OK, continuing to filter")
        return "continue"
    
    def _should_retry_reranking(self, state: WorkflowState) -> Literal["retry", "end"]:
        """
        Decision: Should we retry with broader retrieval after reranking?
        
        Retry if:
        - Reranking confidence is below threshold
        - Haven't exceeded max retries
        - Using focused strategy (can broaden)
        """
        confidence = state.get("reranker_confidence", 0.0)
        retry_count = state.get("retry_count", 0)
        strategy = state.get("strategy", "focused")
        
        if (confidence < self.MIN_RERANKER_CONFIDENCE and 
            retry_count < self.MAX_RETRY_COUNT and 
            strategy == "focused"):
            logger.warning(f"  → Reranker confidence {confidence:.2f} < {self.MIN_RERANKER_CONFIDENCE}, will retry")
            return "retry"
        
        logger.info(f"  → Reranker confidence {confidence:.2f} OK, workflow complete")
        return "end"
    
    def _calculate_retrieval_confidence(self, results: List[Dict]) -> float:
        """
        Calculate confidence score for retrieval results.
        
        Factors:
        - Number of results retrieved
        - Score of top result (higher is better)
        - Score distribution (steep drop-off indicates clear winner)
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not results or len(results) < 3:
            return 0.3  # Low confidence if too few results
        
        # Factor 1: Number of results (normalized)
        count_score = min(len(results) / 50, 1.0) * 0.3
        
        # Factor 2: Top result score (raw hybrid score, roughly in [0, 1] before domain boosts)
        top_score = results[0].get("raw_score", results[0].get("final_score", 0.0))
        score_normalized = min(top_score, 1.0) * 0.4
        
        # Factor 3: Score distribution (check if top results are clearly better)
        if len(results) >= 5:
            top_3_avg = sum(r.get("final_score", 0) for r in results[:3]) / 3
            next_5_avg = sum(r.get("final_score", 0) for r in results[3:8]) / 5
            distribution_score = min((top_3_avg - next_5_avg) / 0.2, 1.0) * 0.3 if next_5_avg > 0 else 0.15
        else:
            distribution_score = 0.15
        
        confidence = count_score + score_normalized + distribution_score
        return min(confidence, 1.0)
    
    def _calculate_reranking_confidence(self, results: List[Dict], query_info: Dict) -> float:
        """
        Calculate confidence score for reranked results.
        
        Factors:
        - Role/skill match with query info
        - Diversity of assessment types in top results
        - Number of high-quality results
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not results or len(results) < 3:
            return 0.4  # Low confidence if too few results
        
        # Factor 1: Check if top results match query role/skills
        role = query_info.get("role", "").lower() if query_info else ""
        skills = [s.lower() for s in query_info.get("skills", [])] if query_info else []
        
        role_skill_matches = 0
        for result in results[:5]:
            name = result.get("name", "").lower()
            desc = result.get("description", "").lower()
            
            if role and role in name:
                role_skill_matches += 2  # Role match is strong signal
            
            for skill in skills[:3]:  # Check top 3 skills
                if skill in name or skill in desc:
                    role_skill_matches += 1
                    break
        
        match_score = min(role_skill_matches / 10, 1.0) * 0.5
        
        # Factor 2: Diversity of test types in top results
        test_types = set()
        for result in results[:5]:
            tt_field = result.get("test_types", "")
            if isinstance(tt_field, list):
                for tt in tt_field:
                    if tt:
                        test_types.add(str(tt).split(",")[0].strip().lower())
            elif isinstance(tt_field, str):
                if tt_field:
                    test_types.add(tt_field.split(",")[0].strip().lower())
            else:
                # Unknown type, skip
                continue
        
        diversity_score = min(len(test_types) / 3, 1.0) * 0.3
        
        # Factor 3: Number of results
        count_score = min(len(results) / 10, 1.0) * 0.2
        
        confidence = match_score + diversity_score + count_score
        return min(confidence, 1.0)
    
    async def run(self, query: str, custom_test_type_ratio: Optional[Dict[str, float]] = None) -> Dict:
        """
        Execute the workflow for a given query.
        
        Args:
            query: User query string
            custom_test_type_ratio: Optional custom K/P ratio (e.g., {"K": 0.6, "P": 0.4})
            
        Returns:
            Dictionary containing:
                - results: Final reranked assessment recommendations
                - state: Complete workflow state for debugging
                - confidence: Final confidence scores
        """
        logger.info("="*80)
        logger.info("Starting LangGraph Workflow Orchestration")
        logger.info(f"Query: {query[:80]}...")
        if custom_test_type_ratio:
            logger.info(f"Custom test type ratio: {custom_test_type_ratio}")
        logger.info("="*80)
        
        # Initialize state
        initial_state: WorkflowState = {
            "query": query,
            "enhanced_query": None,
            "query_info": None,
            "retrieval_results": None,
            "retrieval_confidence": 0.0,
            "reranked_results": None,
            "reranker_confidence": 0.0,
            "retry_count": 0,
            "strategy": "focused",
            "error": None,
            "custom_test_type_ratio": custom_test_type_ratio  # type: ignore
        }
        
        degraded_reasons = start_request()
        
        # Execute graph
        try:
            # LangGraph's compiled graph uses ainvoke for async execution
            final_state = await self.graph.ainvoke(initial_state)  # type: ignore
            
            results = final_state.get("reranked_results", [])
            
            logger.info("="*80)
            logger.info("Workflow Complete")
            logger.info(f"  Final Results: {len(results)} assessments")
            logger.info(f"  Retrieval Confidence: {final_state.get('retrieval_confidence', 0):.2f}")
            logger.info(f"  Reranker Confidence: {final_state.get('reranker_confidence', 0):.2f}")
            logger.info(f"  Retry Count: {final_state.get('retry_count', 0)}")
            logger.info(f"  Strategy Used: {final_state.get('strategy', 'N/A')}")
            logger.info("="*80)
            
            return {
                "results": results,
                "state": final_state,
                "confidence": {
                    "retrieval": final_state.get("retrieval_confidence", 0),
                    "reranking": final_state.get("reranker_confidence", 0)
                },
                "degraded_reasons": list(degraded_reasons),
                "error": final_state.get("error")
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return {
                "results": [],
                "state": initial_state,
                "confidence": {"retrieval": 0.0, "reranking": 0.0},
                "degraded_reasons": list(degraded_reasons),
                "error": str(e)
            }


# Singleton instance
_orchestrator = None

def get_orchestrator() -> WorkflowOrchestrator:
    """Get or create the workflow orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = WorkflowOrchestrator()
    return _orchestrator


async def run_workflow_graph(query: str) -> List[Dict]:
    """
    Convenience function to run workflow and return results.
    
    Args:
        query: User query string
        
    Returns:
        List of recommended assessments
    """
    orchestrator = get_orchestrator()
    result = await orchestrator.run(query)
    return result["results"]


if __name__ == "__main__":
    # Test the workflow
    async def test():
        from src.logging_config import setup_logging
        setup_logging(level="INFO")
        
        test_query = "I am hiring for Java developers who can also collaborate effectively with my business teams."
        
        result = await run_workflow_graph(test_query)
        
        print("\n" + "="*80)
        print("TOP 5 RESULTS:")
        print("="*80)
        for i, assessment in enumerate(result[:5], 1):
            print(f"{i}. {assessment['name']}")
            print(f"   Duration: {assessment.get('duration', 'N/A')}")
            print(f"   Type: {assessment.get('test_types', 'N/A')}")
            print()
    
    asyncio.run(test())
