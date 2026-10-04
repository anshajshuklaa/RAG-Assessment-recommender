from src.gemini_client import generate
import os
import json
from typing import Dict
from dotenv import load_dotenv
from src.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

class HybridQueryEnhancer:
    """
    Hybrid query enhancement combining static skill expansions with LLM-based extraction.
    
    This class enhances user queries by:
    1. Extracting roles, skills, and preferences using Gemini LLM
    2. Expanding skills using predefined mappings and synonyms
    3. Applying static fallback when LLM extraction is insufficient
    4. Adding domain-specific skill expansions based on role/industry keywords
    
    The enhancement process maximizes assessment matching recall by creating
    comprehensive keyword-rich queries from natural language job descriptions.
    """

    # Static JSON expansions derived from your assessment data
    STATIC_SKILL_EXPANSIONS = {
    "java": [
        "spring", "spring boot", "sql", "maven", "hibernate", "junit", "microservices", "java 8", "java 11",
        "oop", "rest api"
    ],
    "python": [
        "django", "flask", "fastapi", "pandas", "numpy", "scikit-learn", "pytest", "machine learning",
        "automation", "api", "data science"
    ],
    "javascript": [
        "react", "angular", "vue", "node.js", "typescript", "npm", "webpack", "frontend", "ui", "ux"
    ],
    "bank admin": [
        "clerical", "data entry", "office administration", "customer service", "document management",
        "accounting", "compliance", "risk management"
    ],
    "leadership": [
        "management", "strategy", "decision making", "team building", "executive", "operations",
        "project management", "communication", "conflict resolution"
    ],
    "data analyst": [
        "sql", "data visualization", "statistical analysis", "reporting", "excel", "power bi",
        "tableau", "data modeling"
    ],
    "devops": [
        "docker", "kubernetes", "ci/cd", "jenkins", "aws", "terraform", "monitoring", "cloud infrastructure"
    ],
    "qa": [
        "testing", "automation", "selenium", "junit", "mockito", "pytest", "test planning"
    ],
    "hr": [
        "talent acquisition", "employee engagement", "payroll", "labor law", "performance management"
    ],
    "marketing": [
        "campaign management", "market research", "brand strategy", "digital marketing", 
        "content strategy", "analytics", "seo", "social media", "email marketing", 
        "marketing automation", "customer segmentation"
    ],
    "sales": [
        "cold calling", "account management", "sales presentations", "objection handling", 
        "territory planning", "pipeline management", "lead generation", "customer relationship", 
        "negotiation", "closing deals", "crm", "prospecting"
    ],
    # Executive/Leadership roles
    "coo": [
        "operational excellence", "p&l management", "strategic planning", "business operations", 
        "process optimization", "change management", "performance optimization", "supply chain",
        "business strategy", "organizational development", "leadership", "executive",
        "personality assessment", "opq", "situational judgment", "behavioral"
    ],
    "ceo": [
        "executive leadership", "strategic planning", "vision", "business strategy", 
        "stakeholder management", "corporate governance", "decision making",
        "personality assessment", "opq", "leadership", "situational judgment"
    ],
    "vp": [
        "leadership", "strategic planning", "team management", "executive", 
        "business development", "operations", "departmental oversight",
        "personality assessment", "opq", "situational judgment", "behavioral"
    ],
    "director": [
        "leadership", "management", "strategic planning", "team building", 
        "project management", "budget management", "cross-functional collaboration",
        "personality assessment", "opq", "situational judgment"
    ],
    "manager": [
        "team management", "leadership", "project management", "communication", 
        "performance management", "decision making", "resource allocation",
        "personality assessment", "opq", "situational judgment"
    ],
    # Admin/Office roles
    "admin": [
        "office management", "scheduling", "data entry", "documentation", 
        "customer service", "record keeping", "microsoft office", "coordination",
        "calendar management", "travel arrangements", "expense reporting"
    ],
    "assistant": [
        "administrative support", "scheduling", "calendar management", "documentation", 
        "office management", "coordination", "time management", "data entry",
        "meeting coordination", "correspondence"
    ],
    "clerical": [
        "data entry", "filing", "documentation", "office administration", 
        "record keeping", "customer service", "organizational skills"
    ],
    # Content/Writing roles
    "content writer": [
        "writing", "seo", "copywriting", "content strategy", "editing", 
        "grammar", "storytelling", "creative writing", "content marketing",
        "cms", "drupal", "wordpress", "content management",  # CMS skills
        "search engine optimization", "digital marketing",  # SEO variants
        "opq", "personality assessment", "work style", "written english"  # Personality for cultural fit
    ],
    "writer": [
        "writing", "editing", "grammar", "storytelling", "communication", 
        "research", "content creation", "proofreading"
    ],
    "copywriter": [
        "copywriting", "advertising", "persuasive writing", "brand voice", 
        "marketing copy", "creative writing", "seo"
    ],
    # Consultant/Advisory roles - ENHANCED FOR BETTER MATCHING
    "consultant": [
        # Core consultant skills that match expected assessments
        "numerical reasoning", "verbal reasoning", "analytical thinking",
        "problem solving", "professional", "cognitive assessment",
        "critical thinking", "logical reasoning", "verify", "inductive reasoning",
        "professional judgment", "business acumen", "deductive reasoning",
        "numerical calculation", "verbal ability", "communication", 
        "administrative", "professional skills", "personality assessment",
        "opq", "personality questionnaire", "situational judgment",
        # Traditional consultant skills  
        "client management", "business analysis", "research", 
        "presentation skills", "project management", "stakeholder management", 
        "consulting", "strategic planning", "change management", "process improvement",
        # Assessment-specific terms that should boost relevant tests
        "verify interactive", "verify g+", "professional solution"
    ],
    "consulting": [
        "analytical", "problem solving", "numerical", "verbal",
        "strategic thinking", "client management", "business acumen",
        "professional", "cognitive", "reasoning", "verify", "inductive"
    ],
    # Add specific professional roles mapping
    "professional": [
        "professional skills", "business skills", "workplace competencies", 
        "professional development", "analytical thinking", "communication", 
        "numerical reasoning", "administrative skills"
    ],
    # Additional technical roles
    "frontend": [
        "html", "css", "javascript", "react", "ui", "ux", "responsive design", "web development"
    ],
    "backend": [
        "api", "database", "server", "rest api", "sql", "microservices", "authentication"
    ],
    "fullstack": [
        "frontend", "backend", "database", "api", "javascript", "react", "node.js", "sql"
    ],
    # Enhanced technical skills for better matching
    "automation": [
        "selenium", "test automation", "qa automation", "automated testing", 
        "test frameworks", "continuous integration", "devops"
    ],
    "testing": [
        "manual testing", "automated testing", "quality assurance", "test cases", 
        "bug tracking", "regression testing", "qa", "selenium"
    ],
    "sql": [
        "database", "mysql", "postgresql", "data analysis", "queries", 
        "database design", "stored procedures", "data manipulation"
    ],
    "r programming": [
        "statistical analysis", "data science", "machine learning", "analytics", 
        "statistical computing", "data visualization", "statistical modeling"
    ],
    # Banking and finance specific
    "banking": [
        "financial services", "compliance", "risk management", "customer service", 
        "banking operations", "loan processing", "account management"
    ],
    "finance": [
        "financial analysis", "accounting", "numerical reasoning",
        "excel", "spreadsheet", "financial modeling", "budgeting",
        "financial reporting", "bookkeeping", "audit", "investment", 
        "risk assessment", "quantitative analysis", "balance sheet", "p&l"
    ],
    "financial": [
        "accounting", "numerical", "finance", "excel", "bookkeeping",
        "financial analysis", "budgeting", "financial reporting"
    ],
    "accounting": [
        "financial", "bookkeeping", "audit", "numerical", "excel",
        "ledger", "financial statements", "reconciliation"
    ],
    "finance analyst": [
        "financial analysis", "numerical reasoning", "accounting",
        "excel", "spreadsheet", "business acumen", "financial modeling",
        "budgeting", "forecasting", "quantitative analysis"
    ],
    "financial analyst": [
        "financial modeling", "excel", "accounting", "budgeting", "forecasting",
        "numerical reasoning", "quantitative analysis", "financial reporting",
        "valuation", "investment analysis", "financial statements", "data analysis"
    ],
    "operations analyst": [
        "process optimization", "data analysis", "numerical reasoning",
        "business acumen", "excel", "project management", "analytical thinking"
    ],
    "analyst": [
        "analysis", "numerical reasoning", "problem solving",
        "excel", "data analysis", "critical thinking", "analytical thinking"
    ],
    "accountant": [
        "accounting", "bookkeeping", "financial reporting", "ledger", "excel",
        "tax", "audit", "compliance", "gaap", "financial statements", "reconciliation"
    ],
    # Industry-specific enhancements
    "retail": [
        "customer service", "sales", "inventory management", "pos systems", 
        "merchandising", "retail operations"
    ],
    "hospitality": [
        "customer service", "guest relations", "operations management", 
        "service excellence", "hospitality management"
    ],
    # Assessment type indicators
    "cognitive": [
        "problem solving", "critical thinking", "analytical skills", "reasoning", 
        "logical thinking", "decision making"
    ],
    "personality": [
        "behavioral assessment", "personality traits", "work style", "cultural fit", 
        "behavioral tendencies", "interpersonal skills"
    ],
    "technical skills": [
        "programming", "coding", "software development", "technical knowledge", 
        "technical assessment", "skills test"
    ],
    # Seniority level indicators
    "mid-level": ["intermediate", "mid", "3-5 years", "experienced"],
    "senior": ["senior", "lead", "5+ years", "7+ years", "expert"],
    "entry-level": ["junior", "entry", "graduate", "0-2 years", "fresher"]
}

    # Skill synonym mappings to improve matching (e.g., "digital marketing" → "digital advertising")
    SKILL_SYNONYMS = {
        "digital marketing": ["digital marketing", "digital advertising", "online marketing", "internet marketing"],
        "online marketing": ["digital marketing", "digital advertising", "online marketing", "internet marketing"],
        "programming": ["programming", "software development", "coding", "software engineering"],
        "software development": ["programming", "software development", "coding", "software engineering"],
        "communication": ["communication", "interpersonal communications", "interpersonal skills", "verbal communication"],
        "interpersonal": ["communication", "interpersonal communications", "interpersonal skills", "verbal communication"],
        "email writing": ["email writing", "business writing", "professional writing", "written communication"],
        "excel": ["excel", "microsoft excel", "spreadsheets", "data analysis"],
        "verify": ["verify", "shl verify", "cognitive assessment", "reasoning test"],
        "inductive": ["inductive reasoning", "logical reasoning", "pattern recognition", "analytical thinking"],
        "verbal": ["verbal ability", "verbal reasoning", "language comprehension", "reading comprehension"],
        "numerical": ["numerical reasoning", "numerical ability", "quantitative reasoning", "math skills"],
        # Content Writer specific synonyms (improves recall for content-focused roles)
        "copywriting": ["copywriting", "content writing", "writing", "editing", "seo writing"],
        "seo": ["seo", "search engine optimization", "digital marketing", "content optimization"],
        "cms": ["cms", "content management system", "drupal", "wordpress", "web content management"],
        "content writer": ["content writer", "copywriter", "writer", "content creator", "content strategist"]
    }


    def __init__(self):
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("Set GEMINI_API_KEY in .env")

    def _expand_with_synonyms(self, skills: list) -> list:
        """
        Expand skills with synonyms to improve matching.
        E.g., 'digital marketing' also searches for 'digital advertising'
        """
        expanded = []
        for skill in skills:
            expanded.append(skill)  # Keep original
            skill_lower = skill.lower()
            # Check if this skill has synonyms
            for key, synonyms in self.SKILL_SYNONYMS.items():
                if key in skill_lower:
                    expanded.extend(synonyms)
        return list(set(expanded))  # Remove duplicates

    def _static_expand(self, query: str):
        """Fallback: Static keyword matching from predefined dictionary"""
        query_lower = query.lower()
        expansions = []
        for keyword, related_skills in self.STATIC_SKILL_EXPANSIONS.items():
            if keyword in query_lower:
                expansions.extend(related_skills)
        return list(set(expansions))

    def enhance(self, query: str) -> Dict:
        """
        Enhance query by extracting and expanding role, skills, and preferences.
        
        This method implements a multi-step enhancement strategy:
        1. LLM extraction: Use Gemini to extract structured information from query
        2. Static fallback: Apply predefined expansions if LLM extraction is insufficient
        3. Concatenation: Combine all extracted fields into comprehensive search query
        
        Args:
            query: Natural language job description or role query
            
        Returns:
            Dictionary containing:
                - enhanced: Enhanced query string for retrieval
                - all_skills: List of all extracted and expanded skills
                - role: Extracted role/job title
                - seniority: Detected seniority level (if any)
        """
        
        # PRIMARY: LLM extraction with inference
        llm_data = self._extract_top_keywords_llm(query)
        
        if llm_data:
            role = llm_data.get("role", "")
            skills = llm_data.get("skills", [])
            preferences = llm_data.get("preferences", [])
            test_types = llm_data.get("test_types", [])
            seniority = llm_data.get("seniority", "")
            focus_area = llm_data.get("focus_area", "")
        else:
            role = ""
            skills = []
            preferences = []
            test_types = []
            seniority = ""
            focus_area = ""
        
        # FALLBACK: If LLM fails or returns empty, use static expansions
        if not skills:
            logger.warning("LLM returned no skills, falling back to static expansions")
            skills = self._static_expand(query)
        
        # SYNONYM EXPANSION: Expand skills with synonyms (e.g., "digital marketing" → "digital advertising")
        expanded_skills = self._expand_with_synonyms(skills)
        
        # Build enhanced query using extracted keywords only (not full job description)
        # This approach prevents query dilution and improves matching precision
        # Target: 10-20 key terms for optimal retrieval performance
        all_terms = []
        if role:
            all_terms.append(role)
        all_terms.extend(expanded_skills)  # Use expanded skills with synonyms
        all_terms.extend(preferences)
        all_terms.extend(test_types)
        
        if all_terms:
            # Use only extracted keywords for focused retrieval
            enhanced_query = ' '.join(all_terms)
        else:
            # Fallback to original if extraction failed
            enhanced_query = query
        
        return {
            "original": query,
            "enhanced": enhanced_query,
            "extracted_skills": skills,
            "inferred_skills": skills,  # All skills are inferred by LLM
            "preferences": preferences,  # NEW: Assessment preferences
            "test_types": test_types,    # NEW: Assessment types
            "role": role,
            "seniority": seniority,
            "focus_area": focus_area,
            "all_skills": skills
        }
    
    def _extract_top_keywords_llm(self, query: str) -> Dict:
        """
        Extract structured information from job description using LLM.
        
        Extracts role, skills, preferences, test types, seniority, and focus area
        with intelligent inference to match appropriate assessment types for both
        technical and non-technical roles.
        
        Args:
            query: Job description or role query text
            
        Returns:
            Dictionary containing extracted fields, or None if extraction fails
        """
        prompt = f"""You are an expert HR assessment specialist. Extract and infer structured information from this job description.

TASK: Extract these fields with intelligent inference:
1. role - Job title/function (e.g., "Java Developer", "Sales Manager")
2. skills - 5-8 core technical + soft skills required
3. preferences - Assessment preferences (coding, behavioral, remote, adaptive, etc.)
4. test_types - Assessment types needed (coding tests, behavioral, personality, etc.)
5. seniority - Level (entry/mid/senior/executive/graduate)
6. focus_area - Primary domain (technical/sales/leadership/clerical/etc.)

INFERENCE RULES (Extract from context even if not explicit):

FOR TECHNICAL ROLES (5-6 core skills + soft skills if mentioned):
- "Java Developer": skills=["Java", "Spring Boot", "SQL", "REST API", "Microservices"], preferences=["coding", "technical"], test_types=["coding assessments", "technical tests"]
- "Java Developer + collaboration": skills=["Java", "Spring Boot", "SQL", "Collaboration", "Communication", "Teamwork"], preferences=["coding", "behavioral"], test_types=["coding assessments", "communication tests"]
- "Data Analyst": skills=["SQL", "Excel", "Python", "Statistics", "Visualization"], preferences=["analytical", "numerical"], test_types=["data analysis", "SQL tests"]
- "QA Engineer": skills=["Selenium", "Automation Testing", "Manual Testing", "SQL"], preferences=["technical", "automation"], test_types=["QA assessments", "automation tests"]
- "Frontend Developer": skills=["JavaScript", "React", "HTML", "CSS", "UI/UX"], preferences=["coding", "frontend"], test_types=["frontend coding", "UI tests"]

FOR NON-TECHNICAL ROLES (6-7 core skills only):
- "Sales": skills=["Cold Calling", "Negotiation", "CRM", "Account Management", "Pipeline Management"], preferences=["behavioral", "situational"], test_types=["sales scenarios", "personality tests"]
- "Marketing": skills=["Campaign Management", "Brand Strategy", "Digital Marketing", "Analytics"], preferences=["creative", "analytical"], test_types=["marketing scenarios", "case studies"]
- "COO/Executive": skills=["Strategy", "P&L Management", "Operations", "Leadership", "Change Management"], preferences=["leadership", "strategic"], test_types=["leadership assessment", "business acumen"]
- "Admin/Clerical": skills=["Data Entry", "Microsoft Office", "Scheduling", "Organization"], preferences=["clerical", "numerical"], test_types=["data entry tests", "clerical speed"]
- "Consultant": skills=["Problem Solving", "Business Analysis", "Client Management", "Research"], preferences=["analytical", "consulting"], test_types=["case studies", "problem-solving tests"]
- "HR": skills=["Recruitment", "Employee Relations", "Talent Development"], preferences=["behavioral", "interpersonal"], test_types=["HR scenarios", "behavioral tests"]
- "Content Writer": skills=["Copywriting", "SEO", "Editing", "Grammar"], preferences=["writing", "creative"], test_types=["writing tests", "grammar tests"]

IMPORTANT: If the job description doesn't explicitly list all fields, INFER them based on the role. For example:
- Technical roles should include "coding assessments" in test_types
- Sales/Marketing roles should include "behavioral", "situational judgment" in test_types
- Leadership roles should include "leadership assessment", "strategic thinking" in test_types

SKILL INFERENCE & EXPANSION:
1. ABBREVIATION EXPANSION:
   - "ML" → "Machine Learning"
   - "AI" → "Artificial Intelligence"
   - "NLP" → "Natural Language Processing"
   - "CV" → "Computer Vision"
   - "OOP" → "Object-Oriented Programming"
   - "API" → "REST API", "API Development"
   - "DB" → "Database", "SQL"
   - "CI/CD" → "Continuous Integration", "Continuous Deployment"

2. ROLE-BASED INFERENCE (add even if not mentioned):
   - "Python Developer" → ALWAYS add ["pytest", "git", "debugging", "unit testing"]
   - "Java Developer" → ALWAYS add ["Maven", "JUnit", "Git", "debugging"]
   - "React Developer" → ALWAYS add ["Next.js", "TypeScript", "npm", "webpack"]
   - "Data Scientist" → ALWAYS add ["pandas", "numpy", "jupyter", "visualization"]
   - "DevOps Engineer" → ALWAYS add ["Docker", "Kubernetes", "monitoring", "scripting"]
   - "Full Stack" → ALWAYS add ["Git", "REST API", "Database", "Testing"]

3. SOFT SKILLS EXTRACTION:
   - If query mentions "collaborate", "collaboration", "teamwork" → ADD: ["Collaboration", "Communication", "Teamwork"]
   - If query mentions "communicate", "communication" → ADD: ["Communication", "Interpersonal Skills"]
   - If query mentions "leadership", "lead", "manage" → ADD: ["Leadership", "Management"]
   - If query mentions "problem solving", "analytical" → ADD: ["Problem Solving", "Analytical Thinking"]
   - Include BOTH technical AND soft skills when both are mentioned in query

4. FRAMEWORK INFERENCE (add common frameworks for languages):
   - "Java" → add "Spring Boot" (most common)
   - "Python" → add "Django" or "Flask" (web roles)
   - "JavaScript" → add "React" or "Node.js" (frontend/backend)
   - ".NET" → add "C#", "ASP.NET"

CRITICAL: Return 5-7 most important skills (include soft skills if mentioned), 2-3 preferences, 2-3 test_types.
Balance technical skills with soft skills when both are required.

Return ONLY valid JSON:
{{
  "role": "detected job role",
  "seniority": "entry/mid/senior/executive",
  "focus_area": "primary domain",
  "skills": ["skill1", "skill2", "skill3", "skill4", "skill5"],
  "preferences": ["preference1", "preference2"],
  "test_types": ["test_type1", "test_type2"]
}}

JOB DESCRIPTION:
\"\"\"{query}\"\"\"
"""
        try:
            text = generate(prompt).strip()
            
            # Remove code blocks if any
            if '```' in text:
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            
            data = json.loads(text)
            
            # Limit fields to optimal sizes
            skills = data.get("skills", [])[:8]  # Max 8 skills
            preferences = data.get("preferences", [])[:4]  # Max 4 preferences
            test_types = data.get("test_types", [])[:4]  # Max 4 test types
            
            return {
                "role": data.get("role", ""),
                "seniority": data.get("seniority", ""),
                "focus_area": data.get("focus_area", ""),
                "skills": skills,
                "preferences": preferences,
                "test_types": test_types
            }
        except Exception as e:
            logger.warning(f"LLM keyword extraction failed: {e}")
            return {
                "role": "",
                "seniority": "",
                "focus_area": "",
                "skills": [],
                "preferences": [],
                "test_types": []
            }


# Test module
if __name__ == "__main__":
    from src.logging_config import setup_logging
    setup_logging(level="INFO")
    
    enhancer = HybridQueryEnhancer()
    queries = [
        "Senior Java Developer",
        "Bank Admin ICICI",
        "COO China",
        "Python Developer with machine learning",
        "Frontend React Developer"
    ]
    for q in queries:
        result = enhancer.enhance(q)
        logger.info(f"Original: {q}")
        logger.info(f"Enhanced: {result['enhanced']}")
        logger.info(f"Skills: {result['all_skills']}")
        logger.info(f"Role: {result['role']}, Seniority: {result['seniority']}")
