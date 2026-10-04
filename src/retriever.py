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
from src.gemini_client import EMBEDDING_DIM, EMBEDDING_MODEL, embed
from dotenv import load_dotenv
import faiss

load_dotenv()

logger = logging.getLogger(__name__)

def parse_duration_minutes(value) -> Optional[int]:
    """Parse catalogue durations like '49 minutes' into an int; None when unknown."""
    match = re.search(r"\d+", str(value)) if pd.notna(value) else None
    return int(match.group()) if match else None


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
    
    def __init__(
        self,
        embeddings_path: str = "outputs/embeddings_gemini_001.npy",
        faiss_index_path: str = "outputs/faiss_gemini_001.index",
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
            ValueError: If GEMINI_API_KEY not found in environment
            FileNotFoundError: If required index files are missing
        """
        logger.info("Initializing Hybrid Retriever")
        
        if not os.getenv("GEMINI_API_KEY"):
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
        
        logger.info("Hybrid Retriever initialization complete")
    
    def _expand_query_for_bm25(self, query: str) -> str:
        """Return query without expansion (expansion reduced accuracy)."""
        return query
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate a normalised query embedding (EMBEDDING_DIM dimensions)."""
        try:
            return embed(query, task_type="RETRIEVAL_QUERY")[0]
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            logger.warning("Falling back to zero vector - retrieval quality will be degraded")
            report_degraded("embedding_unavailable")
            return np.zeros(EMBEDDING_DIM)
    
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
    
    def retrieve(self, query: str, k: int = 10, return_scores: bool = False):
        """
        Retrieve top-k assessments using hybrid scoring.
        
        Combines semantic search, keyword matching, specificity scoring, and quality
        filtering to rank assessments by relevance.
        
        Args:
            query: Enhanced query text (role + skills + preferences + test types)
            k: Number of results to return (default: 10)
            return_scores: If True, return (index, score_dict) tuples instead of just indices
            
        Returns:
            If return_scores=False: List of assessment indices sorted by relevance (highest first)
            If return_scores=True: List of (index, score_dict) tuples with detailed scores
            
        Scoring formula:
            score = 0.3*semantic + 0.2*bm25 + 0.4*specificity + 0.1*quality
        """
        # Compute individual score components
        query_emb = self._get_query_embedding(query)
        semantic_scores = self._get_dense_scores(query_emb)
        bm25_scores = self._get_sparse_scores(query)
        specificity_scores = self._get_specificity_scores(query)
        
        # Quality filtering
        quality_scores = np.array([self._calculate_quality_score(self.df.iloc[i]) for i in range(len(self.df))])
        quality_scores = quality_scores / quality_scores.max()
        
        # Weighted combination
        hybrid_scores = (
            self.semantic_weight * semantic_scores +
            self.bm25_weight * bm25_scores +
            self.specificity_weight * specificity_scores +
            self.QUALITY_WEIGHT * quality_scores
        )
        
        # Return top-k indices
        top_indices = np.argsort(hybrid_scores)[::-1][:k].tolist()
        
        if return_scores:
            # Return indices with detailed scores
            results = []
            for idx in top_indices:
                score_dict = {
                    'semantic': float(semantic_scores[idx]),
                    'bm25': float(bm25_scores[idx]),
                    'specificity': float(specificity_scores[idx]),
                    'quality': float(quality_scores[idx]),
                    'hybrid': float(hybrid_scores[idx])
                }
                results.append((idx, score_dict))
            return results
        else:
            return top_indices
    
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