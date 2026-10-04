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
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
import pandas as pd
from src.gemini_client import EMBEDDING_DIM, EMBEDDING_MODEL, embed
from dotenv import load_dotenv
import faiss

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
        
        Applies domain-specific boosts for technical terms, roles, and assessment types.
        This component carries the highest weight (40%) in the hybrid scoring formula.
        
        Args:
            query: Query text (lowercase comparison)
            
        Returns:
            Normalized specificity scores in range [0, 1]
            
        Note:
            Special handling for functional manager roles (e.g., "Programming Manager")
            prioritizes functional skills over generic management assessments.
        """
        query_lower = query.lower()
        
        # Technical and domain keywords for exact matching
        technical_keywords = [
            # Programming languages
            'java', 'python', 'javascript', 'c++', 'c#', 'sql', 'r', 'php', 'ruby', 'go', 'kotlin', 'swift',
            # Frameworks/Tools
            'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring', 'selenium', 'junit', 'pytest',
            'tableau', 'excel', 'power bi', 'sap', 'oracle', 'aws', 'azure', 'docker', 'kubernetes',
            # Roles (distinguish consultant from manager)
            'developer', 'engineer', 'qa', 'tester', 'analyst', 'manager', 'director', 'coo', 'ceo',
            'admin', 'assistant', 'sales', 'marketing', 'leader', 'leadership',
            # Finance & Accounting (CRITICAL FIX)
            'finance', 'financial', 'accounting', 'accountant', 'bookkeeping', 'budgeting', 'forecasting',
            # Consultant-specific patterns (separate from generic management)
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
        
        # Consultant-specific assessment keywords
        consultant_assessment_keywords = [
            'numerical calculation', 'administrative professional', 'verbal ability', 
            'personality questionnaire', 'opq32r', 'professional international',
            'verify interactive', 'verify verbal', 'short form'
        ]
        
        # Seniority indicators for role-level matching
        seniority_keywords = ['entry', 'senior', 'lead', 'principal', 'staff', 'graduate', 'junior']
        
        # Extract keywords present in query
        present_keywords = [kw for kw in technical_keywords if kw in query_lower]
        present_seniority = [kw for kw in seniority_keywords if kw in query_lower]
        present_consultant_assessments = [kw for kw in consultant_assessment_keywords if kw in query_lower]
        
        # Detect query patterns for specialized scoring
        is_consultant_query = 'consultant' in query_lower
        is_programming_manager = any(term in query_lower for term in ['programming manager', 'software manager', 'development manager'])
        is_marketing_manager = any(term in query_lower for term in ['marketing manager', 'brand manager', 'campaign manager'])
        is_content_writer = any(term in query_lower for term in ['content writer', 'copywriter', 'content creator'])
        is_data_analyst = any(term in query_lower for term in ['data analyst', 'business analyst', 'data scientist'])
        is_finance_analyst = any(term in query_lower for term in ['finance', 'financial', 'accounting', 'accountant', 'finance analyst', 'financial analyst'])
        
        # Initialize specificity scores
        specificity_scores = np.zeros(len(self.df))
        
        # Pre-compute lowercase fields for efficiency
        names_lower = self.df['name'].str.lower().fillna('').values
        descs_lower = self.df['description'].str.lower().fillna('').values
        
        # Score constants
        FUNCTIONAL_SKILL_BOOST = 10.0
        COGNITIVE_ASSESSMENT_BOOST = 8.0
        TOOL_ASSESSMENT_BOOST = 7.0
        GENERIC_ROLE_PENALTY = -3.0
        CONSULTANT_PROFESSIONAL_BOOST = 5.0
        CONSULTANT_MGMT_PENALTY = -2.0
        KEYWORD_NAME_BOOST = 3.0
        KEYWORD_DESC_BOOST = 0.2
        SENIORITY_BOOST = 4.0
        
        for i in range(len(self.df)):
            name_lower = names_lower[i]
            desc_lower = descs_lower[i]
            
            # Functional manager role scoring: prioritize functional skills over generic management
            if is_programming_manager or is_marketing_manager:
                functional_skills = ['marketing', 'programming', 'digital advertising', 'campaign', 'brand']
                cognitive_assessments = ['verify', 'inductive', 'verbal', 'reasoning', 'cognitive']
                tool_assessments = ['excel', 'microsoft', 'sql', 'powerpoint', 'office']
                
                if any(skill in name_lower for skill in functional_skills):
                    specificity_scores[i] += FUNCTIONAL_SKILL_BOOST
                if any(assess in name_lower for assess in cognitive_assessments):
                    specificity_scores[i] += COGNITIVE_ASSESSMENT_BOOST
                if any(tool in name_lower for tool in tool_assessments):
                    specificity_scores[i] += TOOL_ASSESSMENT_BOOST
                if 'manager solution' in name_lower or 'manager 8' in name_lower:
                    specificity_scores[i] += GENERIC_ROLE_PENALTY
            
            # Content writer role scoring
            if is_content_writer:
                writer_boosts = {
                    'seo': 9.0,
                    'cms': 8.0,
                    'grammar': 7.0,
                    'personality': 6.0
                }
                writer_keywords = {
                    'seo': ['seo', 'search engine'],
                    'cms': ['drupal', 'wordpress', 'cms', 'content management'],
                    'grammar': ['grammar', 'written english', 'writing', 'english comprehension'],
                    'personality': ['opq', 'personality', 'work style']
                }
                
                for category, keywords in writer_keywords.items():
                    if any(kw in name_lower for kw in keywords):
                        specificity_scores[i] += writer_boosts[category]
            
            # Data analyst role scoring
            if is_data_analyst:
                analyst_keywords = {
                    'python': (['python', 'pandas', 'numpy'], 8.0),
                    'sql': (['sql', 'database', 'data warehousing'], 8.0),
                    'excel': (['excel', 'spreadsheet'], 7.0),
                    'numerical': (['numerical', 'quantitative', 'reasoning'], 6.0)
                }
                
                for keywords, boost in analyst_keywords.values():
                    if any(kw in name_lower for kw in keywords):
                        specificity_scores[i] += boost
                        break
            
            # Finance analyst role scoring - CRITICAL FIX
            if is_finance_analyst:
                finance_keywords = {
                    'financial': (['financial', 'finance', 'accounting'], 10.0),
                    'excel': (['excel', 'spreadsheet', 'microsoft excel'], 9.0),
                    'numerical': (['numerical', 'calculation', 'quantitative'], 8.0),
                    'accounting': (['accounting', 'bookkeeping', 'ledger'], 8.0),
                    'budgeting': (['budget', 'forecasting', 'planning'], 7.0),
                    'analytical': (['analytical', 'analysis', 'reasoning'], 6.0)
                }
                
                for keywords, boost in finance_keywords.values():
                    if any(kw in name_lower for kw in keywords):
                        specificity_scores[i] += boost
                        break
                
                # PENALTY: Strongly penalize customer service/contact center for finance queries
                if any(term in name_lower for term in ['customer service', 'contact center', 'call center', 'customer support']):
                    specificity_scores[i] -= 10.0  # Strong penalty to push these down
            
            # Consultant query scoring: prefer analytical over management
            if is_consultant_query:
                if any(term in name_lower for term in ['professional', 'numerical', 'verbal', 'administrative', 'calculation', 'verify', 'opq']):
                    specificity_scores[i] += CONSULTANT_PROFESSIONAL_BOOST
                if any(term in name_lower for term in ['manager', 'management', 'leadership', 'director']):
                    specificity_scores[i] += CONSULTANT_MGMT_PENALTY
            
            # Consultant-specific assessment matches
            for kw in present_consultant_assessments:
                if kw in name_lower:
                    specificity_scores[i] += 6.0
                elif kw in desc_lower:
                    specificity_scores[i] += 2.0
            
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
    
    def _apply_domain_boosts(self, query: str, specificity_scores: np.ndarray) -> np.ndarray:
        """Apply domain-specific boosts for finance, analyst, and consultant queries."""
        query_lower = query.lower()
        names_lower = self.df['name'].str.lower().fillna('').values
        
        # FINANCE DOMAIN BOOST
        finance_keywords = ["finance", "financial", "accounting", "bookkeeping", "operations analyst"]
        if any(kw in query_lower for kw in finance_keywords):
            logger.info("Finance domain detected - applying boosts")
            for i in range(len(self.df)):
                name_lower = names_lower[i]
                
                # Very strong boost for financial assessments
                if any(kw in name_lower for kw in ["financial", "accounting", "bookkeeping", "audit", "budgeting"]):
                    specificity_scores[i] += 15.0  # Increased from 10.0
                    logger.debug(f"Finance boost: +15.0 for '{self.df.iloc[i]['name']}'")
                # Strong boost for numerical/excel (common in finance)
                elif any(kw in name_lower for kw in ["numerical", "excel", "spreadsheet", "quantitative"]):
                    specificity_scores[i] += 12.0  # Increased from 10.0
                    logger.debug(f"Finance numerical boost: +12.0 for '{self.df.iloc[i]['name']}'")
                # Moderate boost for business acumen
                elif "business acumen" in name_lower:
                    specificity_scores[i] += 8.0
                    logger.debug(f"Finance business boost: +8.0 for '{self.df.iloc[i]['name']}'")
                
                # Strong penalty for generic "Professional" or "Solution" assessments in finance context
                if any(kw in name_lower for kw in ["professional +", "professional solution", "manager solution", "director solution"]):
                    # Only penalize if it doesn't have finance-specific keywords
                    if not any(kw in name_lower for kw in ["financial", "accounting", "numerical", "excel"]):
                        specificity_scores[i] -= 8.0  # Strong penalty
                        logger.debug(f"Finance generic penalty: -8.0 for '{self.df.iloc[i]['name']}'")
                
                # Penalize customer service/contact center for finance queries
                if any(kw in name_lower for kw in ["customer service", "contact center", "data entry", "call center"]):
                    specificity_scores[i] -= 10.0  # Increased from 5.0
                    logger.debug(f"Finance penalty: -10.0 for '{self.df.iloc[i]['name']}'")
        
        # ANALYST DOMAIN BOOST
        analyst_keywords = ["analyst", "analysis"]
        if any(kw in query_lower for kw in analyst_keywords):
            logger.info("Analyst domain detected - applying boosts")
            for i in range(len(self.df)):
                name_lower = names_lower[i]
                
                # Boost analytical/professional assessments
                if any(kw in name_lower for kw in ["numerical", "analytical", "problem solving", "critical thinking", "cognitive", "inductive", "verify"]):
                    specificity_scores[i] += 8.0  # Increased from 6.0
                    logger.debug(f"Analyst boost: +8.0 for '{self.df.iloc[i]['name']}'")
                # Moderate boost for professional/personality
                elif any(kw in name_lower for kw in ["professional", "opq", "personality"]):
                    specificity_scores[i] += 5.0
                    logger.debug(f"Analyst professional boost: +5.0 for '{self.df.iloc[i]['name']}'")
                
                # Penalize generic "Professional +" solutions for analyst queries
                if any(kw in name_lower for kw in ["professional +", "professional solution"]):
                    # Only penalize if it doesn't have analyst-specific keywords
                    if not any(kw in name_lower for kw in ["numerical", "analytical", "cognitive", "verify"]):
                        specificity_scores[i] -= 5.0
                        logger.debug(f"Analyst generic penalty: -5.0 for '{self.df.iloc[i]['name']}'")
                
                # Penalize entry-level/data entry for mid/senior analyst queries
                if any(kw in name_lower for kw in ["data entry", "entry-level", "clerk"]):
                    if any(kw in query_lower for kw in ["mid", "senior", "experienced", "3+", "5+", "7+"]):
                        specificity_scores[i] -= 4.0  # Increased from 3.0
                        logger.debug(f"Analyst seniority penalty: -4.0 for '{self.df.iloc[i]['name']}'")
        
        # CONSULTANT DOMAIN BOOST
        consultant_keywords = ["consultant", "consulting"]
        if any(kw in query_lower for kw in consultant_keywords):
            logger.info("Consultant domain detected - applying boosts")
            for i in range(len(self.df)):
                name_lower = names_lower[i]
                
                # Strong boost for verify/cognitive assessments
                if any(kw in name_lower for kw in ["verify", "inductive", "deductive", "cognitive"]):
                    specificity_scores[i] += 10.0
                    logger.debug(f"Consultant verify boost: +10.0 for '{self.df.iloc[i]['name']}'")
                # Boost for numerical/verbal/analytical
                elif any(kw in name_lower for kw in ["numerical", "verbal", "analytical", "professional", "reasoning"]):
                    specificity_scores[i] += 8.0
                    logger.debug(f"Consultant analytical boost: +8.0 for '{self.df.iloc[i]['name']}'")
                # Boost for OPQ/personality
                elif any(kw in name_lower for kw in ["opq", "personality", "situational"]):
                    specificity_scores[i] += 6.0
                    logger.debug(f"Consultant personality boost: +6.0 for '{self.df.iloc[i]['name']}'")
                
                # Penalize generic management solutions for consultant
                if "manager solution" in name_lower or "director solution" in name_lower:
                    specificity_scores[i] -= 3.0
                    logger.debug(f"Consultant penalty: -3.0 for '{self.df.iloc[i]['name']}'")
        
        # EXECUTIVE/COO DOMAIN BOOST
        executive_keywords = ["coo", "ceo", "executive", "vp", "vice president", "chief"]
        if any(kw in query_lower for kw in executive_keywords):
            logger.info("Executive domain detected - applying boosts")
            for i in range(len(self.df)):
                name_lower = names_lower[i]
                
                # Strong boost for leadership/personality assessments
                if any(kw in name_lower for kw in ["opq", "personality", "leadership", "executive"]):
                    specificity_scores[i] += 12.0
                    logger.debug(f"Executive personality boost: +12.0 for '{self.df.iloc[i]['name']}'")
                # Boost for situational/behavioral
                elif any(kw in name_lower for kw in ["situational", "behavioral", "judgment", "scenarios"]):
                    specificity_scores[i] += 10.0
                    logger.debug(f"Executive situational boost: +10.0 for '{self.df.iloc[i]['name']}'")
                # Boost for cognitive/reasoning
                elif any(kw in name_lower for kw in ["verify", "cognitive", "reasoning", "inductive"]):
                    specificity_scores[i] += 7.0
                    logger.debug(f"Executive cognitive boost: +7.0 for '{self.df.iloc[i]['name']}'")
        
        return specificity_scores
    
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
        
        # Apply domain-specific boosts (CRITICAL FIX for finance/analyst/consultant)
        specificity_scores = self._apply_domain_boosts(query, specificity_scores)
        
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