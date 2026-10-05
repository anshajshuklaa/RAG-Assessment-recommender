"""
Hybrid Retriever Module

Implements a hybrid retrieval system combining semantic search, BM25 keyword matching,
and specificity scoring for SHL assessment recommendations.

Architecture:
    - Semantic Search: 30% weight (Gemini gemini-embedding-001 + FAISS)
    - BM25 Matching: 20% weight (keyword-based retrieval)
    - Specificity Scoring: 40% weight (exact keyword matching with domain boosts)
    - Quality Filtering: 10% weight (assessment quality indicators)
"""

import logging
import numpy as np
import pickle
import re
import os
from typing import List, Dict, Optional
import pandas as pd
from src.degradation import report_degraded
from src.gemini_client import EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_PROVIDER, embed, index_path
from dotenv import load_dotenv
import faiss

load_dotenv()

logger = logging.getLogger(__name__)

def parse_duration_minutes(value) -> Optional[int]:
    """Parse catalogue durations like '49 minutes' into an int; None when unknown."""
    match = re.search(r"\d+", str(value)) if pd.notna(value) else None
    return int(match.group()) if match else None


def split_query(query: str, min_words: int = 60, chunk_words: int = 30, max_chunks: int = 6) -> List[str]:
    """
    Split a long job description into the full text plus a few focused sub-queries.

    Long pasted JDs mix the role with boilerplate (company blurb, benefits); retrieving per
    chunk and fusing the rankings keeps one strong signal from being drowned by the rest.
    Short queries are returned unchanged.
    """
    words = query.split()
    if len(words) < min_words:
        return [query]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+|\n+", query) if s.strip()]
    chunks, current = [], []
    for sentence in sentences:
        current.extend(sentence.split())
        if len(current) >= chunk_words:
            chunks.append(" ".join(current))
            current = []
    if len(current) >= 5:
        chunks.append(" ".join(current))
    return [query] + chunks[:max_chunks]


class HybridRetriever:
    """
    Hybrid retrieval system for SHL assessment recommendations.
    
    Combines multiple retrieval strategies:
        - Semantic search using Gemini embedding-001 and FAISS
        - BM25 keyword matching for exact term relevance
        - Specificity scoring with domain-specific boosts
        - Quality filtering to prioritize comprehensive assessments
    
    Attributes:
        embeddings: NumPy array of document embeddings (backup)
        faiss_index: FAISS index for fast similarity search
        bm25: BM25 model for keyword-based retrieval
        df: DataFrame containing assessment metadata
        semantic_weight: Weight for semantic search component (0.3)
        bm25_weight: Weight for BM25 component (0.2)
        specificity_weight: Weight for specificity component (0.4)
    """
    
    # Retrieval weights optimized through empirical evaluation
    SEMANTIC_WEIGHT = 0.3
    BM25_WEIGHT = 0.2
    SPECIFICITY_WEIGHT = 0.4
    QUALITY_WEIGHT = 0.1
    COMPONENT_WEIGHTS = {
        'semantic': SEMANTIC_WEIGHT,
        'bm25': BM25_WEIGHT,
        'specificity': SPECIFICITY_WEIGHT,
        'quality': QUALITY_WEIGHT,
    }
    RRF_K = 60  # standard RRF constant; dampens the gap between neighbouring ranks
    # Two-stage shortlist: the first HEAD_K results keep the weights above; the rest of the
    # shortlist (what the reranker sees) comes from a semantic-heavy ranking, which finds the
    # "companion" tests that share no words with the query. See upgrade-log.md (two-stage shortlist).
    HEAD_K = 10
    TAIL_SEMANTIC_WEIGHT = 0.7
    
    def __init__(
        self,
        embeddings_path: str = "outputs/embeddings_gemini_001.npy",
        faiss_index_path: Optional[str] = None,
        bm25_path: str = "outputs/bm25_index.pkl",
        assessments_path: str = "outputs/assessments_processed.csv",
    ):
        """
        Initialize hybrid retriever with pre-computed indices.
        
        Args:
            embeddings_path: Path to Gemini embedding-001 vectors
            faiss_index_path: Path to FAISS index file
            bm25_path: Path to pickled BM25 model
            assessments_path: Path to assessments CSV
            
        Raises:
            ValueError: If GEMINI_API_KEY is missing while EMBEDDING_PROVIDER=gemini
            FileNotFoundError: If required index files are missing
        """
        logger.info("Initializing Hybrid Retriever")
        
        if EMBEDDING_PROVIDER == "gemini" and not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY not found in environment. Please set it in .env file")
        
        # Load embeddings (fallback for FAISS)
        logger.info("Loading embedding vectors")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path, allow_pickle=True)
            logger.info(f"Loaded embeddings with shape: {self.embeddings.shape}")
        else:
            logger.warning(f"Embeddings not found at {embeddings_path}")
            logger.warning("Run 'python migrate_to_gemini_001_faiss.py' to generate embeddings")
            self.embeddings = None
        
        # Load FAISS index
        logger.info("Loading FAISS index")
        faiss_index_path = faiss_index_path or index_path()
        if os.path.exists(faiss_index_path):
            self.faiss_index = faiss.read_index(faiss_index_path)
            logger.info(f"Loaded FAISS IndexFlatIP with {self.faiss_index.ntotal} vectors")
            if self.faiss_index.d != EMBEDDING_DIM:
                raise ValueError(
                    f"FAISS index dimension {self.faiss_index.d} != {EMBEDDING_DIM} for {EMBEDDING_MODEL}. "
                    "Rebuild it with 'python scripts/build_index.py'"
                )
        else:
            logger.warning(f"FAISS index not found at {faiss_index_path}")
            logger.warning("Falling back to NumPy dot product if embeddings available")
            self.faiss_index = None
        
        # Load BM25 index
        logger.info("Loading BM25 index")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(
                f"BM25 index not found at {bm25_path}. "
                "This file is required for the application to run. "
                "Please ensure all required files are present in the outputs/ directory."
            )
        with open(bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)
        logger.info("BM25 index loaded successfully")
        
        # Load assessments
        logger.info("Loading assessment database")
        if not os.path.exists(assessments_path):
            raise FileNotFoundError(
                f"Assessments file not found at {assessments_path}. "
                "This file is required for the application to run. "
                "Please ensure all required files are present in the outputs/ directory."
            )
        self.df = pd.read_csv(assessments_path)
        self.df["duration_minutes"] = self.df["duration"].map(parse_duration_minutes)
        logger.info(f"Loaded {len(self.df)} assessments")
        
        # Set retrieval weights
        self.semantic_weight = self.SEMANTIC_WEIGHT
        self.bm25_weight = self.BM25_WEIGHT
        self.specificity_weight = self.SPECIFICITY_WEIGHT
        self.fusion = os.getenv("RETRIEVAL_FUSION", "weighted")
        self.decompose = os.getenv("RETRIEVAL_DECOMPOSE", "false").lower() in {"1", "true", "yes"}
        
        # Quality scores depend only on the catalogue, so compute them once
        quality = np.array([self._calculate_quality_score(self.df.iloc[i]) for i in range(len(self.df))])
        self._quality_scores = quality / quality.max()
        
        logger.info("Hybrid Retriever initialization complete")
    
    def _expand_query_for_bm25(self, query: str) -> str:
        """Return query without expansion (expansion reduced accuracy)."""
        return query
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate a normalised query embedding (EMBEDDING_DIM dimensions)."""
        return self._get_query_embeddings([query])[0]

    def _get_query_embeddings(self, queries: List[str]) -> np.ndarray:
        """Embed several queries in one call; zero vectors (and a degraded flag) on failure."""
        try:
            return embed(queries, task_type="RETRIEVAL_QUERY")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            logger.warning("Falling back to zero vector - retrieval quality will be degraded")
            report_degraded("embedding_unavailable")
            return np.zeros((len(queries), EMBEDDING_DIM))
    
    def _get_dense_scores(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute semantic similarity scores using FAISS vector search."""
        if self.faiss_index is not None:
            # Use FAISS for fast similarity search
            query_vec = query_embedding.reshape(1, -1).astype('float32')
            similarities, ids = self.faiss_index.search(query_vec, self.faiss_index.ntotal)
            # FAISS returns results sorted by similarity; scatter them back to catalogue row order
            scores = np.empty(self.faiss_index.ntotal, dtype="float32")
            scores[ids[0]] = similarities[0]
        elif self.embeddings is not None:
            # Fallback to NumPy dot product
            scores = np.dot(self.embeddings, query_embedding)
        else:
            logger.error("No embeddings or FAISS index available")
            return np.zeros(len(self.df))
        
        # Normalize to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores
    
    def _get_sparse_scores(self, query: str) -> np.ndarray:
        """Compute BM25 keyword matching scores."""
        # Expand query with synonyms for better matching (NEW)
        expanded_query = self._expand_query_for_bm25(query)
        
        tokens = re.findall(r'\b\w+\b', expanded_query.lower())
        scores = self.bm25.get_scores(tokens)
        
        # Normalize to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores
    
    def _get_specificity_scores(self, query: str) -> np.ndarray:
        """
        Compute specificity scores based on exact keyword matching.
        
        Rewards assessments whose name/description contain query keywords (technical terms,
        roles, assessment types) and seniority terms. No role-specific rules.
        
        Args:
            query: Query text (lowercase comparison)
            
        Returns:
            Normalized specificity scores in range [0, 1]
        """
        query_lower = query.lower()
        
        # Technical and domain keywords for exact matching
        technical_keywords = [
            # Programming languages
            'java', 'python', 'javascript', 'c++', 'c#', 'sql', 'r', 'php', 'ruby', 'go', 'kotlin', 'swift',
            # Frameworks/Tools
            'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring', 'selenium', 'junit', 'pytest',
            'tableau', 'excel', 'power bi', 'sap', 'oracle', 'aws', 'azure', 'docker', 'kubernetes',
            # Roles
            'developer', 'engineer', 'qa', 'tester', 'analyst', 'manager', 'director', 'coo', 'ceo',
            'admin', 'assistant', 'sales', 'marketing', 'leader', 'leadership',
            # Finance & Accounting
            'finance', 'financial', 'accounting', 'accountant', 'bookkeeping', 'budgeting', 'forecasting',
            'consultant', 'consulting', 'advisory', 'professional services',
            # Core skills assessments
            'communication', 'verbal', 'numerical', 'inductive', 'reasoning', 'cognitive', 'personality',
            'seo', 'content', 'writing', 'english', 'data entry', 'customer service',
            'calculation', 'administrative', 'professional', 'verify', 'interactive', 'opq', 'questionnaire',
            # Assessment types (from test_types field)
            'coding', 'technical', 'behavioral', 'situational', 'judgment', 'personality', 'leadership',
            'analytical', 'case study', 'scenarios', 'problem solving', 'business acumen',
            # Preferences
            'adaptive', 'remote', 'creative', 'strategic', 'clerical', 'detail-oriented'
        ]
        
        # Seniority indicators for role-level matching
        seniority_keywords = ['entry', 'senior', 'lead', 'principal', 'staff', 'graduate', 'junior']
        
        # Extract keywords present in query
        present_keywords = [kw for kw in technical_keywords if kw in query_lower]
        present_seniority = [kw for kw in seniority_keywords if kw in query_lower]
        
        # Initialize specificity scores
        specificity_scores = np.zeros(len(self.df))
        
        # Pre-compute lowercase fields for efficiency
        names_lower = self.df['name'].str.lower().fillna('').values
        descs_lower = self.df['description'].str.lower().fillna('').values
        
        # Score constants (generic signals only: no role-specific rules)
        KEYWORD_NAME_BOOST = 3.0
        KEYWORD_DESC_BOOST = 0.2
        SENIORITY_BOOST = 4.0
        
        for i in range(len(self.df)):
            name_lower = names_lower[i]
            desc_lower = descs_lower[i]
            
            # Exact keyword matches using word boundaries
            for kw in present_keywords:
                pattern = re.compile(rf"\b{re.escape(kw)}\b")
                if pattern.search(name_lower):
                    specificity_scores[i] += KEYWORD_NAME_BOOST
                elif pattern.search(desc_lower):
                    specificity_scores[i] += KEYWORD_DESC_BOOST
            
            # Seniority level matches
            for kw in present_seniority:
                pattern = re.compile(rf"\b{re.escape(kw)}\b")
                if pattern.search(name_lower):
                    specificity_scores[i] += SENIORITY_BOOST
        
        # Normalize to [0, 1]
        if specificity_scores.max() > 0:
            specificity_scores = specificity_scores / specificity_scores.max()
        
        return specificity_scores
    
    def _calculate_quality_score(self, assessment_row) -> float:
        """
        Calculate quality score for assessment prioritization.
        
        Applies bonuses for comprehensive, modern, and feature-rich assessments.
        
        Args:
            assessment_row: Row from assessments DataFrame
            
        Returns:
            Quality score in range [0.1, ~2.0]
            
        Quality indicators:
            - "Solution" assessments: comprehensive evaluation suites (+0.3)
            - "New" versions: updated content and improved validity (+0.2)
            - "Interactive" format: engaging user experience (+0.2)
            - Duration 45-90 min: optimal for thorough assessment (+0.1)
            - Adaptive IRT: personalized difficulty adjustment (+0.1)
            - Remote testing: modern requirement support (+0.05)
        """
        score = 1.0
        name = assessment_row['name'].lower()
        
        # Comprehensive assessment suites
        if 'solution' in name:
            score += 0.3
            
        # Updated versions
        if 'new' in name or '(new)' in name:
            score += 0.2
            
        # Modern interactive format
        if 'interactive' in name:
            score += 0.2
            
        # Optimal duration range
        duration_str = str(assessment_row.get('duration', ''))
        if 'minutes' in duration_str.lower():
            try:
                minutes = int(''.join(filter(str.isdigit, duration_str)))
                if 45 <= minutes <= 90:
                    score += 0.1
                elif minutes >= 30:
                    score += 0.05
            except ValueError:
                pass
        
        # Penalty for highly specialized assessments
        specialized_indicators = ['short form', 'short-form', 'sift out', 'report only']
        if any(indicator in name for indicator in specialized_indicators):
            score -= 0.1
        
        # Advanced features
        if assessment_row.get('adaptive_irt_support') == 'Yes':
            score += 0.1
        if assessment_row.get('remote_testing_support') == 'Yes':
            score += 0.05
            
        return max(score, 0.1)
    
    def _component_scores(self, query: str, query_emb: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-component scores in [0, 1] for one (sub-)query."""
        return {
            'semantic': self._get_dense_scores(query_emb),
            'bm25': self._get_sparse_scores(query),
            'specificity': self._get_specificity_scores(query),
            'quality': self._quality_scores,
        }

    def _weighted_fusion(self, components: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted sum of [0, 1] component scores."""
        return sum(self.COMPONENT_WEIGHTS[name] * scores for name, scores in components.items())

    def _rrf_fusion(self, components: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Weighted Reciprocal Rank Fusion: sum_c w_c / (RRF_K + rank_c(d)).

        Uses ranks only, so it is insensitive to how each component's scores are scaled.
        Normalised so an item ranked first by every component scores 1.0.
        """
        fused = np.zeros(len(self.df))
        for name, scores in components.items():
            if not np.any(scores):
                continue  # e.g. zero-vector semantic scores when embeddings are unavailable
            ranks = np.empty(len(scores))
            ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
            fused += self.COMPONENT_WEIGHTS[name] / (self.RRF_K + ranks)
        return fused / (sum(self.COMPONENT_WEIGHTS.values()) / (self.RRF_K + 1))

    def retrieve(
        self,
        query: str,
        k: int = 10,
        return_scores: bool = False,
        fusion: Optional[str] = None,
        decompose: Optional[bool] = None,
    ):
        """
        Retrieve top-k assessments using hybrid scoring.
        
        Combines semantic search, keyword matching, specificity scoring, and quality
        filtering to rank assessments by relevance.
        
        Args:
            query: Enhanced query text (role + skills + preferences + test types)
            k: Number of results to return (default: 10)
            return_scores: If True, return (index, score_dict) tuples instead of just indices
            fusion: "weighted" (default) or "rrf"; overrides RETRIEVAL_FUSION
            decompose: split long queries into sub-queries and fuse their rankings;
                overrides RETRIEVAL_DECOMPOSE (default off)
            
        Returns:
            If return_scores=False: List of assessment indices sorted by relevance (highest first)
            If return_scores=True: List of (index, score_dict) tuples with detailed scores
        """
        fusion = fusion or self.fusion
        decompose = self.decompose if decompose is None else decompose
        fuse = self._rrf_fusion if fusion == "rrf" else self._weighted_fusion

        sub_queries = split_query(query) if decompose else [query]
        embeddings = self._get_query_embeddings(sub_queries)

        # Score the full query and each sub-query, then combine the per-query rankings with RRF
        per_query = [fuse(self._component_scores(q, e)) for q, e in zip(sub_queries, embeddings)]
        if len(per_query) == 1:
            hybrid_scores = per_query[0]
        else:
            hybrid_scores = self._rrf_fusion_queries(per_query)
        
        top_indices = np.argsort(-hybrid_scores, kind="stable")[:k].tolist()
        if fusion == "weighted" and len(per_query) == 1 and k > self.HEAD_K:
            top_indices = self._fill_tail(top_indices, self._component_scores(query, embeddings[0]), k)
        
        if return_scores:
            components = self._component_scores(query, embeddings[0])
            results = []
            for idx in top_indices:
                score_dict = {name: float(scores[idx]) for name, scores in components.items()}
                score_dict['hybrid'] = float(hybrid_scores[idx])
                results.append((idx, score_dict))
            return results
        else:
            return top_indices

    def _fill_tail(self, ranked: List[int], components: Dict[str, np.ndarray], k: int) -> List[int]:
        """Keep the top HEAD_K results; fill the rest of the shortlist from a semantic-heavy ranking."""
        if not np.any(components['semantic']):
            return ranked  # embeddings unavailable: nothing to add
        rest = 1 - self.TAIL_SEMANTIC_WEIGHT
        others = sum(w for name, w in self.COMPONENT_WEIGHTS.items() if name != 'semantic')
        tail_scores = sum(
            (self.TAIL_SEMANTIC_WEIGHT if name == 'semantic' else w / others * rest) * components[name]
            for name, w in self.COMPONENT_WEIGHTS.items()
        )
        head = ranked[:self.HEAD_K]
        seen = set(head)
        tail = [int(i) for i in np.argsort(-tail_scores, kind="stable") if i not in seen]
        return head + tail[:k - len(head)]

    def _rrf_fusion_queries(self, per_query: List[np.ndarray]) -> np.ndarray:
        """Equal-weight RRF across sub-query rankings, normalised to 1.0 for an item ranked first everywhere."""
        fused = np.zeros(len(self.df))
        for scores in per_query:
            ranks = np.empty(len(scores))
            ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
            fused += 1.0 / (self.RRF_K + ranks)
        return fused / (len(per_query) / (self.RRF_K + 1))
    
    def get_candidates(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve candidate assessments with full metadata.
        
        Args:
            query: Query text
            k: Number of candidates to return (default: 10)
            
        Returns:
            List of dicts containing assessment details:
                - index: DataFrame row index
                - name: Assessment name
                - description: Full description
                - url: Assessment URL
                - duration: Estimated completion time
                - test_types: List of assessment types
        """
        indices = self.retrieve(query, k=k)
        
        candidates = []
        for idx in indices:
            row = self.df.iloc[idx]
            candidates.append({
                'index': idx,
                'name': row['name'],
                'description': row['description'],
                'url': row['url'],
                'duration': row.get('duration', 'N/A'),
                'test_types': row.get('test_types', '[]'),
            })
        
        return candidates


def main():
    """Test retriever with sample query."""
    retriever = HybridRetriever()
    
    query = "Python developer with Django experience"
    logger.info(f"Testing query: {query}")
    
    candidates = retriever.get_candidates(query, k=5)
    
    logger.info(f"Top {len(candidates)} results:")
    for i, c in enumerate(candidates, 1):
        logger.info(f"{i}. {c['name']}")
        logger.info(f"   URL: {c['url']}")


if __name__ == "__main__":
    main()