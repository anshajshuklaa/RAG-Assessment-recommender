"""
Intelligent reranking module using Gemini 2.0 Flash.

Provides assessment reranking based on sophisticated matching criteria
including role-skill alignment, seniority matching, and assessment type appropriateness.
"""

import os
from src.gemini_client import GENERATION_MODEL, generate
from typing import List, Dict, Optional, cast
import time
from src.logging_config import get_logger

logger = get_logger(__name__)


class GeminiReranker:
    """
    Production-grade reranker using Gemini 2.0 Flash.
    
    Features:
    - Multi-criteria ranking (role alignment, seniority, assessment type)
    - Error handling with retry logic
    - Structured evaluation for consistent results
    - Request statistics tracking
    """
    
    RERANKING_PROMPT_TEMPLATE = """You are an expert HR assessment specialist with deep knowledge of SHL assessments. You have {num_candidates} assessment options for a hiring need.

Your task is to analyze the hiring requirements and rank the assessments from MOST to LEAST relevant using sophisticated matching criteria.

HIRING REQUIREMENT:
{query}

AVAILABLE ASSESSMENTS:
{candidates_text}

ADVANCED RANKING CRITERIA (in order of importance):

1. ROLE-SKILL ALIGNMENT (40% weight):
   - Technical roles: Prioritize coding/technical assessments that match the exact technologies
   - Sales roles: Prioritize sales scenarios, behavioral assessments, personality tests
   - Management roles: Prioritize leadership assessments, managerial scenarios
   - Admin/Clerical: Prioritize data entry, clerical speed, office skills tests
   - Consultant roles: Prioritize case studies, problem-solving, analytical tests

2. SKILL DEPTH MATCHING (25% weight):
   - Entry-level: Basic assessments, foundational skills
   - Mid-level: Intermediate assessments, practical application
   - Senior-level: Advanced assessments, architectural thinking, leadership
   - Executive: Strategic thinking, business acumen, leadership reports

3. ASSESSMENT TYPE APPROPRIATENESS (20% weight):
   - Coding roles: Prioritize "programming", "coding", "technical" assessments
   - Behavioral roles: Prioritize "behavioral", "personality", "situational" tests
   - Mixed roles (technical skills plus soft skills): Balance technical + behavioral assessments
   - Cognitive roles: Prioritize "reasoning", "problem-solving", "analytical" tests

4. DURATION PREFERENCES (10% weight):
   - If query mentions time limit (e.g., "1 hour", "30 minutes"), PREFER (not require) assessments close to that duration
   - Perfect match: Within ±10 minutes of requested duration
   - Good match: Within ±20 minutes
   - Acceptable: Within ±30 minutes
   - For "quick" screening: prefer shorter assessments (15-30 min)
   - For comprehensive evaluation: prefer longer assessments (45-90 min)
   - IMPORTANT: Duration is a PREFERENCE, not a hard requirement - don't exclude assessments just because of duration

5. TECHNOLOGY/DOMAIN SPECIFICITY (5% weight):
   - Exact technology matches (Java, Python, React) get highest priority
   - Related technologies get medium priority (Java → Spring Boot, Python → Django)
   - Generic skills get lower priority

FUNCTIONAL ROLES WITH "MANAGER" IN THE TITLE:
Test the functional skills of the role before generic managerial skills; rank generic "Manager Solution"
assessments below assessments of the role's specific skills.

ASSESSMENT QUALITY INDICATORS (use for tie-breaking):
- "Solution" assessments: Comprehensive, high-quality (prefer over standalone tests)
- "Short Form" assessments: Quick screening versions
- "New" versions: Updated, improved versions (prefer over legacy)
- "Interactive" assessments: Engaging, modern format
- Domain-specific assessments: Industry-tailored (banking, retail, etc.)

OUTPUT FORMAT:
Rank ALL {num_candidates} assessments from 1 (most relevant) to {num_candidates} (least relevant).
Consider ALL criteria above, not just superficial keyword matching.

Return your ranking as a JSON array with the assessment numbers in order of relevance:
[most_relevant_number, second_most_relevant_number, ..., least_relevant_number]

EXAMPLE: For 5 assessments, return [3, 1, 5, 2, 4] where assessment #3 is most relevant and #4 is least relevant.

Do not include any other information in your response, just the JSON array of indices.
"""
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize Gemini reranker.
        
        Uses GEMINI_MODEL (default gemini-3.8-flash) at temperature 0 for
        deterministic rankings.
        
        Args:
            model: Optional model name override. Defaults to GEMINI_MODEL.
        """
        self.model_name = model or GENERATION_MODEL
        self.stats = {
            "reranks": 0,
            "retries": 0,
            "errors": 0
        }
        logger.info(f"Gemini Reranker initialized (model={self.model_name})")
    
    def _format_candidates(self, candidates: List[Dict]) -> str:
        """Format candidates for LLM prompt"""
        text = ""
        for i, cand in enumerate(candidates, 1):
            text += f"{i}. [{cand.get('duration', 'Unknown')}] {cand.get('name', 'Unknown')}\n"
            desc = cand.get('description', '')
            text += f"   {desc[:100] if desc else 'No description'}\n"
            test_types = cand.get('test_types', 'Unknown')
            text += f"   Type: {test_types}\n\n"
        return text
    
    def _parse_response(self, response_text: str, num_candidates: int) -> List[int]:
        """
        Parse Gemini response into list of indices
        Handles JSON array format: [1, 5, 8, 3, ...] or fallback to comma-separated
        """
        import re
        import json
        
        # Try to extract JSON array from response
        match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
        
        try:
            # Try parsing as JSON array
            parsed = json.loads(response_text.strip())
            if isinstance(parsed, list):
                indices = []
                for num in parsed:
                    if isinstance(num, int):
                        idx = num - 1  # Convert to 0-based
                        if 0 <= idx < num_candidates:
                            if idx not in indices:  # Avoid duplicates
                                indices.append(idx)
                if indices:
                    return indices
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: extract numbers from text
        first_line = response_text.strip().split('\n')[0]
        numbers = re.findall(r'\d+', first_line)
        
        indices = []
        for num_str in numbers:
            idx = int(num_str) - 1  # Convert to 0-based
            if 0 <= idx < num_candidates:
                if idx not in indices:  # Avoid duplicates
                    indices.append(idx)
        
        return indices
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10,
        max_retries: int = 2
    ) -> List[Dict]:
        """
        Rerank candidates using Gemini LLM
        
        Args:
            query: Job description
            candidates: List of candidate assessments
            top_k: Number of results to return
            max_retries: Max retry attempts on failure
            
        Returns:
            Reranked list of top-k candidates
        """
        
        # WINNER'S APPROACH: Always rerank even if len(candidates) == top_k
        # Reranking improves the order quality, not just filtering
        if len(candidates) == 0:
            return candidates
        if len(candidates) < top_k:
            top_k = len(candidates)
        
        self.stats["reranks"] += 1
        
        # Format prompt
        candidates_text = self._format_candidates(candidates)
        prompt = self.__class__.RERANKING_PROMPT_TEMPLATE.format(
            query=query,
            candidates_text=candidates_text,
            num_candidates=len(candidates)
        )
        # Debug: print short summary of prompt to help troubleshooting
        print(f"  [reranker] model={self.model_name}, prompt_chars={len(prompt)}, candidates={len(candidates)}")
        
        # Try reranking with retries
        for attempt in range(max_retries + 1):
            try:
                response_text = generate(prompt, model=self.model_name)
                
                # Parse response
                indices = self._parse_response(response_text, len(candidates))
                
                if not indices:
                    raise ValueError("Could not parse response indices")
                
                # Pad with remaining candidates if needed
                if len(indices) < top_k:
                    all_indices = set(range(len(candidates)))
                    used = set(indices)
                    remaining = [i for i in range(len(candidates)) if i not in used]
                    indices.extend(remaining)
                
                # Return reranked candidates
                reranked = [candidates[i] for i in indices[:top_k]]
                return reranked
            
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a quota/rate limit error (429)
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    print(f"  ⚠ API quota exhausted - using fallback ranking")
                    self.stats["errors"] += 1
                    # Don't retry on quota errors - return fallback immediately
                    return candidates[:top_k]
                
                # For other errors, retry
                self.stats["retries"] += 1
                
                if attempt < max_retries:
                    print(f"  Reranking attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)
                else:
                    print(f"  Reranking failed after {max_retries + 1} attempts")
                    self.stats["errors"] += 1
                    # Fallback: return top-k by input order
                    return candidates[:top_k]
        
        return candidates[:top_k]
    
    def get_stats(self) -> Dict:
        """Get reranker statistics"""
        return self.stats.copy()


def main():
    """Test reranker"""
    import pandas as pd
    import pickle
    import numpy as np
    
    # Load data (506 assessments)
    df = pd.read_csv("outputs/assessments_processed.csv")
    embeddings = np.load("outputs/embeddings_gemini_001.npy")  # Updated path
    
    with open("outputs/bm25_index.pkl", 'rb') as f:  # Already updated to 506
        bm25 = pickle.load(f)
    
    # Create sample candidates
    candidates = []
    for i in range(10):
        row = df.iloc[i]
        candidates.append({
            'index': i,
            'name': row['name'],
            'description': row['description'],
            'duration': row.get('duration', 'N/A'),
            'test_types': row.get('test_types', 'Unknown'),
        })
    
    # Test reranking
    reranker = GeminiReranker()
    
    query = "Python developer with Django experience"
    print(f"\nQuery: {query}\n")
    
    reranked = reranker.rerank(query, candidates, top_k=5)
    
    print("Top 5 Reranked Results:")
    for i, cand in enumerate(reranked, 1):
        print(f"{i}. {cand['name']}")
    
    print(f"\nStats: {reranker.get_stats()}")


if __name__ == "__main__":
    main()
