"""
Professional Dark Theme Streamlit UI for SHL Assessment Recommender
Clean, modern, enterprise-grade design
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SHL Assessment Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# Clean White Professional Theme with Animations
# ============================================================================

st.markdown("""
<style>
    /* Clean white base */
    .stApp {
        background: #ffffff;
        color: #2c3e50;
    }
    
    .main {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1rem 2rem;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Professional card style with animations */
    .assessment-card {
        background: #ffffff;
        border: 1px solid #e1e8ed;
        border-left: 4px solid #3498db;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .assessment-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(52, 152, 219, 0.05), transparent);
        transition: left 0.6s ease;
    }
    
    .assessment-card:hover::before {
        left: 100%;
    }
    
    .assessment-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 32px rgba(52, 152, 219, 0.2);
        border-left-color: #e74c3c;
    }
    
    .assessment-card:active {
        transform: translateY(-4px) scale(1.01);
        transition: all 0.1s ease;
    }
    
    /* Status indicators */
    .status-online {
        background: #ffffff;
        color: #27ae60;
        border: 2px solid #27ae60;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.1);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 4px 12px rgba(39, 174, 96, 0.1); }
        50% { box-shadow: 0 4px 20px rgba(39, 174, 96, 0.2); }
        100% { box-shadow: 0 4px 12px rgba(39, 174, 96, 0.1); }
    }
    
    .status-offline {
        background: #ffffff;
        color: #e74c3c;
        border: 2px solid #e74c3c;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(231, 76, 60, 0.1);
    }
    
    /* Animated tags */
    .tag {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .tag:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .tag-knowledge {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
    }
    
    .tag-personality {
        background: linear-gradient(135deg, #9b59b6, #8e44ad);
        color: white;
    }
    
    .tag-duration {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
    }
    
    /* Animated metrics */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e1e8ed;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(52, 152, 219, 0.15);
        border-color: #3498db;
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3498db;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover .metric-number {
        transform: scale(1.1);
        color: #e74c3c;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Animated buttons */
    .stButton button {
        background: #ffffff !important;
        color: #3498db !important;
        border: 2px solid #3498db !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.8rem 1.5rem !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255,255,255,0.2);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton button:hover {
        background: #3498db !important;
        color: white !important;
        border-color: #3498db !important;
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 8px 24px rgba(52, 152, 219, 0.3) !important;
    }
    
    .stButton button:active {
        transform: translateY(-1px) scale(1.02) !important;
        transition: all 0.1s ease !important;
    }
    
    /* Input fields */
    .stTextArea textarea {
        background: #ffffff !important;
        border: 2px solid #e1e8ed !important;
        border-radius: 12px !important;
        color: #2c3e50 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1) !important;
        transform: scale(1.02) !important;
    }
    
    /* Header with animation */
    .header-section {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 2px solid #ecf0f1;
        margin-bottom: 2rem;
        position: relative;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        animation: fadeInUp 1s ease;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin: 0;
        animation: fadeInUp 1s ease 0.2s both;
    }
    
    /* Sidebar with enhanced styling */
    .sidebar-section {
        background: #ffffff;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #3498db, #e74c3c);
        transition: width 0.3s ease;
    }
    
    .sidebar-section:hover {
        transform: translateX(4px);
        box-shadow: 0 8px 24px rgba(52, 152, 219, 0.15);
    }
    
    .sidebar-section:hover::before {
        width: 8px;
    }
    
    /* Template buttons */
    .template-btn {
        background: #ffffff !important;
        border: 2px solid #e1e8ed !important;
        color: #2c3e50 !important;
        border-radius: 10px !important;
        padding: 0.8rem 1rem !important;
        margin-bottom: 0.5rem !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }
    
    .template-btn:hover {
        background: linear-gradient(135deg, #3498db, #2980b9) !important;
        color: white !important;
        transform: translateX(8px) !important;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 650 !important;
        font-size: 1.05rem !important;
        color: #2c3e50 !important;
        letter-spacing: 0.2px;
    }
    
    .streamlit-expanderContent {
        background: #ffffff;
        border-radius: 10px;
        padding-top: 0.5rem;
    }
    
    div[data-testid="stExpander"] * {
        color: #121417 !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="stExpander"] div[data-testid="stSliderLabel"] label,
    div[data-testid="stExpander"] div[data-testid="stSliderMinValue"],
    div[data-testid="stExpander"] div[data-testid="stSliderMaxValue"],
    div[data-testid="stExpander"] div[data-testid="stTickValue"],
    div[data-testid="stExpander"] div[data-testid="stSliderValue"] {
        color: #121417 !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stExpander"] [data-baseweb="slider"] [role="slider"],
    div[data-testid="stExpander"] .stSlider,
    div[data-testid="stExpander"] .stSlider * {
        color: #121417 !important;
        fill: #121417 !important;
    }
    
    div[data-testid="stExpander"] .stSlider .css-16huue1,
    div[data-testid="stExpander"] .stSlider .css-1k0ckh2,
    div[data-testid="stExpander"] .stSlider .css-1dp5vir {
        background: rgba(18, 20, 23, 0.08) !important;
        border-radius: 8px !important;
    }
    
    div[data-testid="stExpander"] .stSlider .css-1k0ckh2 span,
    div[data-testid="stExpander"] .stSlider .css-16huue1 span {
        color: #121417 !important;
    }
    
    div[data-testid="stExpander"] .stSlider .css-1k0ckh2 div {
        color: #121417 !important;
    }
    
    
    /* Links */
    a {
        color: #3498db !important;
        text-decoration: none !important;
        font-weight: 600;
        transition: all 0.3s ease !important;
    }
    
    a:hover {
        color: #e74c3c !important;
        text-decoration: underline !important;
        transform: translateX(4px) !important;
    }
    
    /* Success/Error messages with animation */
    .success-message {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
        animation: slideInRight 0.5s ease;
    }
    
    .error-message {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);
        animation: slideInRight 0.5s ease;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #95a5a6;
        font-size: 0.9rem;
        padding-top: 3rem;
        border-top: 2px solid #ecf0f1;
        margin-top: 3rem;
    }
    
    /* Slider customization */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #3498db, #e74c3c) !important;
    }
    
    /* Checkbox customization */
    .stCheckbox > label > div {
        background: #ffffff !important;
        border: 2px solid #e1e8ed !important;
        border-radius: 6px !important;
    }
    
    .stCheckbox > label > div[data-checked="true"] {
        background: linear-gradient(135deg, #3498db, #2980b9) !important;
        border-color: #3498db !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Helper Functions
# ============================================================================

def call_api(query: str, top_k: int = 10, custom_ratio=None):
    """Call the recommendation API"""
    try:
        payload = {"query": query, "top_k": top_k}
        if custom_ratio:
            payload["test_type_ratio"] = custom_ratio
            
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json=payload,
            headers={"X-API-Key": os.getenv("API_KEY", "")},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# ============================================================================
# Main App
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="header-section">
        <div class="header-title">SHL Assessment Matcher</div>
        <p class="header-subtitle">Professional Talent Assessment Matching Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Status Check
    api_status = check_api_health()
    if not api_status:
        # Center the system offline message
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.markdown(f"""
            <div class="error-message" style="text-align: center;">
                ⚠️ <strong>System Offline</strong> - Cannot reach API at {API_BASE_URL}
            </div>
            """, unsafe_allow_html=True)
        return
    
    # Main Layout - Two columns
    col1, col2 = st.columns([2.5, 1.5], gap="large")
    
    with col1:
        # Job Requirements Section
        st.markdown("### Job Requirements")
        query = st.text_area(
            "Describe the position, required skills, and candidate profile",
            height=120,
            placeholder="Example: Senior Python developer with 5+ years experience in FastAPI, microservices, and AWS. Strong system design skills and ability to mentor junior developers.",
            label_visibility="collapsed",
            value=st.session_state.get("query", "")
        )
        
        # Search Button
        search_clicked = st.button("Find Matching Assessments", use_container_width=True)
    
    with col2:
        # System Status
        if check_api_health():
            st.markdown("""
            <div style="text-align: center; padding: 1rem; margin-bottom: 1rem;">
                <div class="status-online">System Online</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; margin-bottom: 1rem;">
                <div class="status-offline">System Offline</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Settings
        with st.expander("Settings", expanded=False):
            top_k = st.slider(
                "Number of results",
                min_value=5,
                max_value=25,
                value=10,
                help="Maximum number of assessments to return",
                key="top_k_slider"
            )
            st.session_state["top_k"] = top_k
            
            # Custom ratio
            use_custom = st.checkbox("Custom Test Type Ratio", value=False, key="main_custom_ratio")
            st.session_state["use_custom_ratio"] = use_custom
            
            if use_custom:
                k_ratio = st.slider("Knowledge Tests %", 0.0, 1.0, 0.6, 0.05, key="main_k_ratio_slider")
                p_ratio = 1.0 - k_ratio
                st.session_state["k_ratio"] = k_ratio
                st.session_state["p_ratio"] = p_ratio
                st.write(f"Personality Tests: {p_ratio:.0%}")
        
        # Templates
        with st.expander("Quick Templates", expanded=False):
            templates = {
                "Python Developer": "Senior Python developer with FastAPI, microservices, AWS deployment experience",
                "Java Developer": "Java developer with Spring Boot, microservices, team collaboration skills",
                "Data Analyst": "Data analyst with Python, SQL, Excel, and statistical analysis skills",
                "Sales Manager": "Sales manager with team leadership and client relationship experience",
                "DevOps Engineer": "DevOps engineer with Kubernetes, AWS, CI/CD pipeline expertise",
                "Product Manager": "Product manager with technical background and stakeholder management"
            }
            
            for name, desc in templates.items():
                if st.button(name, use_container_width=True, key=name, help=f"Click to use: {desc}"):
                    st.session_state["query"] = desc
                    st.rerun()
        
        # Pro Tips
        with st.expander("Pro Tips", expanded=False):
            st.markdown("""
            <ul style="color: #7f8c8d; font-size: 0.9rem; line-height: 1.6; margin: 0; padding-left: 1rem;">
                <li>Be specific about required skills</li>
                <li>Mention experience level (junior/senior)</li>
                <li>Include soft skills for better matching</li>
                <li>Add industry context when relevant</li>
                <li>Specify team size and collaboration needs</li>
                <li>Include technical stack preferences</li>
            </ul>
            """, unsafe_allow_html=True)
    
    # Results Section
    if search_clicked:
        if not query.strip():
            # Center the error message
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                st.markdown("""
                <div class="error-message" style="text-align: center;">
                    Please provide a job description to get recommendations
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing requirements and matching assessments..."):
                # Get custom ratio if enabled
                custom_ratio = None
                if st.session_state.get("use_custom_ratio", False):
                    custom_ratio = {
                        "K": st.session_state.get("k_ratio", 0.6),
                        "P": st.session_state.get("p_ratio", 0.4)
                    }
                
                result = call_api(query, st.session_state.get("top_k", 10), custom_ratio)
            
            if result and "recommended_assessments" in result:
                assessments = result["recommended_assessments"]
                
                # Success message
                st.markdown(f"""
                <div class="success-message">
                    ✅ Found {len(assessments)} matching assessments
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{len(assessments)}</div>
                        <div class="metric-label">Total Results</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    # Count knowledge tests
                    knowledge_count = 0
                    for a in assessments:
                        test_types = a.get("test_type", [])
                        if isinstance(test_types, list):
                            knowledge_count += sum(1 for t in test_types if 'k' in str(t).lower())
                        else:
                            knowledge_count += 1 if 'k' in str(test_types).lower() else 0
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{knowledge_count}</div>
                        <div class="metric-label">Knowledge Tests</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m3:
                    personality_count = len(assessments) - knowledge_count
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{personality_count}</div>
                        <div class="metric-label">Personality Tests</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Results
                st.markdown("### Assessment Recommendations")
                
                for idx, assessment in enumerate(assessments, 1):
                    name = assessment.get("name", "Unknown Assessment")
                    url = assessment.get("url", "#")
                    duration = assessment.get("duration", "N/A")
                    description = assessment.get("description", "No description available")
                    test_types = assessment.get("test_type", [])
                    
                    # Format test types
                    if isinstance(test_types, list) and test_types:
                        primary_type = str(test_types[0]).upper()
                        type_display = ", ".join(str(t) for t in test_types)
                    else:
                        primary_type = str(test_types).upper() if test_types else "OTHER"
                        type_display = str(test_types) if test_types else "Other"
                    
                    # Determine tag style
                    tag_class = "tag-knowledge" if 'K' in primary_type else "tag-personality"
                    
                    st.markdown(f"""
                    <div class="assessment-card">
                        <h4 style="margin-top: 0; color: #2c3e50;">{idx}. {name}</h4>
                        <p style="color: #7f8c8d; margin: 0.5rem 0;">{description[:150]}{'...' if len(description) > 150 else ''}</p>
                        <div style="margin: 1rem 0;">
                            <span class="tag {tag_class}">{type_display}</span>
                            <span class="tag tag-duration">⏱️ {duration} min</span>
                        </div>
                        <div style="margin-top: 1rem;">
                            <a href="{url}" target="_blank" style="font-weight: 600;">View Assessment Details →</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Export
                st.markdown("---")
                st.markdown("### Export Results")
                
                df = pd.DataFrame(assessments)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV Report",
                    data=csv,
                    file_name=f"shl_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                # Center the no results message
                col_left, col_center, col_right = st.columns([1, 2, 1])
                with col_center:
                    st.markdown("""
                    <div class="error-message" style="text-align: center;">
                        ⚠️ No matching assessments found. Try refining your job description.
                    </div>
                    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>SHL Assessment Matcher</strong> | Enterprise AI Matching Platform</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">FastAPI • Streamlit • Google Gemini • LangGraph</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()