import os
import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field
import json
import re
import requests
import time

# Load environment variables from .env file
load_dotenv()

# Set page config with custom theme
st.set_page_config(
    page_title="Legal Case Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for theming
st.markdown("""
<style>
    div.stButton > button[kind="primary"] {
        background-color: #2C3E50;
        color: white;
    }
    div.stSlider > div[data-baseweb="slider"] > div > div {
        background-color: #34495E !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        border-color: #2C3E50;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f5;
    }
</style>
""", unsafe_allow_html=True)

# Role assignment for different LLMs:
# Deepseek builds defense arguments
# GPT-4o-mini builds prosecution arguments
# GPT-4o makes the final verdict decision

# App title and description
st.title("AI-Powered Legal Case Analyzer")
st.markdown("### Multi-agent legal case analysis system")

# Sidebar for user inputs
st.sidebar.header("Case Parameters")

# Case type selection
case_types = [
    "Criminal - Theft",
    "Criminal - Assault",
    "Criminal - Fraud",
    "Civil - Contract Dispute",
    "Civil - Personal Injury",
    "Civil - Property Dispute",
    "Family - Divorce",
    "Family - Child Custody"
]

selected_case_type = st.sidebar.selectbox(
    "Select Case Type", 
    options=case_types
)

# Case information
st.sidebar.subheader("Case Information")
defendant_name = st.sidebar.text_input("Defendant Name", value="John Doe")
plaintiff_name = st.sidebar.text_input("Plaintiff/Prosecutor Name", value="The State")
case_summary = st.sidebar.text_area(
    "Case Summary", 
    value="The defendant is accused of stealing merchandise worth $5,000 from a retail store.",
    height=150
)

# Case complexity and jurisdiction
st.sidebar.subheader("Case Details")
case_complexity = st.sidebar.select_slider(
    "Case Complexity",
    options=["Simple", "Moderate", "Complex"],
    value="Moderate"
)

jurisdictions = [
    "Federal Court",
    "State Court",
    "Local Court",
    "Appellate Court",
    "Supreme Court"
]

jurisdiction = st.sidebar.selectbox(
    "Jurisdiction", 
    options=jurisdictions,
    index=1
)

# Prior convictions
prior_convictions = st.sidebar.slider("Prior Convictions", 0, 10, 0)

# Evidence strength
evidence_strength = st.sidebar.select_slider(
    "Evidence Strength",
    options=["Weak", "Moderate", "Strong"],
    value="Moderate"
)

# API Keys
openai_api_key = "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"
with st.sidebar.expander("LLM Settings"):
    ollama_url = st.text_input("Ollama Base URL", 
                               value=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
    
    # Ensure the URL has the correct format
    if not ollama_url.endswith('/v1'):
        ollama_url = f"{ollama_url.rstrip('/')}/v1"
        
    ollama_model = st.text_input("Ollama Model", value="deepseek-r1:latest")
    gpt4o_mini_model = st.text_input("GPT-4o Mini Model", value="gpt-4o-mini")
    gpt4o_model = st.text_input("GPT-4o Model", value="gpt-4o")
    use_ollama = st.checkbox("Use Ollama for defense agent", value=True)
    timeout_seconds = st.slider("API Timeout (seconds)", 5, 60, 30)

# Pydantic model for structured output
class CaseVerdict(BaseModel):
    verdict: str = Field(description="Final verdict (Guilty/Not Guilty for criminal cases, or For Plaintiff/For Defendant for civil cases)")
    confidence_score: int = Field(description="Confidence score in the verdict (0-100)")
    reasoning: str = Field(description="Explanation for the verdict")
    sentence_recommendation: str = Field(description="Recommended sentence or judgment")
    appeal_potential: str = Field(description="Assessment of potential for successful appeal")

# Function to test Ollama connection
def test_ollama_connection(base_url, model_name, max_retries=3):
    """Test connection to Ollama server with retries"""
    for attempt in range(max_retries):
        try:
            # Simplest possible request to check if Ollama is responding
            url = f"{base_url.rstrip('/v1')}/api/tags"
            st.write(f"Attempting to connect to Ollama at: {url}")
            
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            # Check if the specified model exists
            models = response.json().get('models', [])
            model_exists = any(model.get('name') == model_name for model in models)
            
            if not model_exists:
                st.warning(f"Model '{model_name}' not found in Ollama. Available models: {[m.get('name') for m in models]}")
                st.info(f"You may need to run: ollama pull {model_name}")
                return False
                
            st.success(f"Successfully connected to Ollama and found model {model_name}")
            return True
            
        except Exception as e:
            st.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                
    st.error(f"Failed to connect to Ollama after {max_retries} attempts")
    return False

# Function to extract JSON from text
def extract_json_from_text(text):
    try:
        # Find JSON pattern with regex
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        # Check if entire text is valid JSON
        return json.loads(text)
    except:
        # If no valid JSON found, return None
        return None

# Test connections button
if st.sidebar.button("Test Connections"):
    st.sidebar.info("Testing OpenAI connection...")
    try:
        test_model = OpenAIModel(
            model_name="gpt-4o-mini",  # Use a simpler model for testing
            provider=OpenAIProvider(
                api_key=openai_api_key,
                timeout=timeout_seconds
            )
        )
        test_agent = Agent(model=test_model, system_prompt="You are a helpful assistant.")
        result = test_agent.run_sync(user_prompt="Say 'Connection successful' in one short sentence.")
        st.sidebar.success("✅ OpenAI connection successful!")
    except Exception as e:
        st.sidebar.error(f"❌ OpenAI connection failed: {str(e)}")
    
    if use_ollama:
        st.sidebar.info("Testing Ollama connection...")
        if test_ollama_connection(ollama_url.rstrip('/v1'), ollama_model):
            try:
                test_model = OpenAIModel(
                    model_name=ollama_model,
                    provider=OpenAIProvider(
                        api_key=openai_api_key,
                        base_url=ollama_url,
                        timeout=timeout_seconds
                    )
                )
                test_agent = Agent(model=test_model, system_prompt="You are a helpful assistant.")
                result = test_agent.run_sync(user_prompt="Say 'Connection successful' in one short sentence.")
                st.sidebar.success("✅ Ollama connection successful!")
            except Exception as e:
                st.sidebar.error(f"❌ Ollama API connection failed: {str(e)}")

# Main analysis function
def run_case_analysis():
    # First, verify OpenAI API key
    if not openai_api_key:
        st.error("⚠️ OpenAI API key is missing. Please add it to your .env file.")
        return
        
    with st.spinner("Initializing AI models..."):
        # First try to set up the models with proper error handling
        defense_model = None
        prosecution_model = None
        verdict_model = None
        
        # Check Ollama connection if needed
        ollama_available = False
        if use_ollama:
            ollama_available = test_ollama_connection(
                base_url=ollama_url.rstrip('/v1'),
                model_name=ollama_model
            )
        
        # Try to set up defense model
        try:
            if use_ollama and ollama_available:
                defense_model = OpenAIModel(
                    model_name=ollama_model,
                    provider=OpenAIProvider(
                        api_key=openai_api_key,
                        base_url=ollama_url,
                        timeout=timeout_seconds
                    )
                )
                st.success("✅ Defense model initialized with Ollama")
            else:
                defense_model = OpenAIModel(
                    model_name=gpt4o_mini_model,
                    provider=OpenAIProvider(
                        api_key=openai_api_key,
                        timeout=timeout_seconds
                    )
                )
                st.success("✅ Defense model initialized with OpenAI")
        except Exception as e:
            st.error(f"❌ Failed to initialize defense model: {str(e)}")
            return
        
        # Set up prosecution model with OpenAI
        try:
            prosecution_model = OpenAIModel(
                model_name=gpt4o_mini_model,
                provider=OpenAIProvider(
                    api_key=openai_api_key,
                    timeout=timeout_seconds
                )
            )
            st.success("✅ Prosecution model initialized")
        except Exception as e:
            st.error(f"❌ Failed to initialize prosecution model: {str(e)}")
            return
            
        # Set up verdict model with OpenAI
        try:
            verdict_model = OpenAIModel(
                model_name=gpt4o_model,
                provider=OpenAIProvider(
                    api_key=openai_api_key,
                    timeout=timeout_seconds
                )
            )
            st.success("✅ Verdict model initialized")
        except Exception as e:
            st.error(f"❌ Failed to initialize verdict model: {str(e)}")
            return

    # Case details formatting
    case_info = f"Case Type: {selected_case_type}\nDefendant: {defendant_name}\nPlaintiff/Prosecutor: {plaintiff_name}\nJurisdiction: {jurisdiction}\nCase Complexity: {case_complexity}\nPrior Convictions: {prior_convictions}\nEvidence Strength: {evidence_strength}\nCase Summary: {case_summary}"
    
    # System prompts
    defense_system_prompt = f"""
    You are an experienced defense attorney representing {defendant_name}.
    Your client is involved in a {selected_case_type} case.
    
    Case Details:
    {case_info}
    
    Your job is to provide the STRONGEST possible defense arguments for your client.
    Focus on legal protections, procedural issues, reasonable doubt, alternative explanations, and mitigating factors.
    Be thorough, professional, and persuasive in your defense.
    """

    prosecution_system_prompt = f"""
    You are an experienced prosecutor or plaintiff's attorney representing {plaintiff_name}.
    You are prosecuting/suing {defendant_name} in a {selected_case_type} case.
    
    Case Details:
    {case_info}
    
    Your job is to provide the STRONGEST possible prosecution/plaintiff arguments.
    Focus on evidence strength, motive, pattern of behavior, legal precedent, and refuting potential defense arguments.
    Be thorough, professional, and persuasive in your arguments.
    """

    verdict_system_prompt = f"""
    You are an impartial judge or jury tasked with determining the verdict for a {selected_case_type} case.
    
    Case Details:
    {case_info}
    
    You have been provided with both defense and prosecution arguments.
    Using ONLY the provided arguments and case details, make a fair and balanced determination.
    Consider the burden of proof appropriate for this type of case.
    
    You MUST structure your response as a valid JSON object with the following fields:
    - verdict (string): "Guilty" or "Not Guilty" for criminal cases, "For Plaintiff" or "For Defendant" for civil cases
    - confidence_score (number): Confidence score in the verdict (0-100)
    - reasoning (string): Detailed explanation for the verdict
    - sentence_recommendation (string): Recommended sentence or judgment if applicable
    - appeal_potential (string): Assessment of potential for successful appeal
    
    Format your JSON response within ```json ``` code blocks.
    """

    # Agent definitions
    defense_agent = Agent(model=defense_model, system_prompt=defense_system_prompt)
    prosecution_agent = Agent(model=prosecution_model, system_prompt=prosecution_system_prompt)
    verdict_agent = Agent(model=verdict_model, system_prompt=verdict_system_prompt)

    # Create tabs for the different analysis sections
    tab1, tab2, tab3 = st.tabs(["Defense Arguments", "Prosecution Arguments", "Final Verdict"])

    # 1. Get defense arguments
    defense_new_messages = []
    with tab1:
        with st.spinner("Generating defense arguments..."):
            st.subheader(f"Defense Arguments for {defendant_name}")
            defense_prompt = f"Present the strongest possible defense arguments for {defendant_name} in this {selected_case_type} case."
            
            try:
                result_defense = defense_agent.run_sync(user_prompt=defense_prompt)
                st.markdown(result_defense.data)
                defense_new_messages = result_defense.new_messages()
            except Exception as e:
                st.error(f"❌ Error generating defense arguments: {str(e)}")
                st.markdown("**Fallback Defense Arguments:**")
                st.markdown(f"Due to a technical error, we were unable to generate detailed defense arguments for {defendant_name}. The case will proceed with basic defense considerations.")

    # 2. Get prosecution arguments
    prosecution_new_messages = []
    with tab2:
        with st.spinner("Generating prosecution arguments..."):
            st.subheader(f"Prosecution Arguments against {defendant_name}")
            prosecution_prompt = f"Present the strongest possible prosecution/plaintiff arguments against {defendant_name} in this {selected_case_type} case."
            
            try:
                result_prosecution = prosecution_agent.run_sync(user_prompt=prosecution_prompt)
                st.markdown(result_prosecution.data)
                prosecution_new_messages = result_prosecution.new_messages()
            except Exception as e:
                st.error(f"❌ Error generating prosecution arguments: {str(e)}")
                st.markdown("**Fallback Prosecution Arguments:**")
                st.markdown(f"Due to a technical error, we were unable to generate detailed prosecution arguments against {defendant_name}. The case will proceed with basic prosecution considerations.")

    # 3. Combine for verdict
    combined_messages_for_verdict = defense_new_messages + prosecution_new_messages

    # 4. Final verdict
    with tab3:
        with st.spinner("Deliberating on final verdict..."):
            st.subheader("Final Verdict")
            
            # Skip verdict if no arguments were generated
            if not defense_new_messages or not prosecution_new_messages:
                st.warning("⚠️ Cannot generate verdict without both defense and prosecution arguments.")
                st.markdown("Please ensure both defense and prosecution arguments were successfully generated before proceeding.")
            else:
                final_question = f"""Based on the arguments presented by both sides, what is your verdict in this {selected_case_type} case? 

Please provide your verdict in JSON format with these fields:
- verdict (string)
- confidence_score (number between 0-100)
- reasoning (string)
- sentence_recommendation (string)
- appeal_potential (string)"""
                
                try:
                    result_verdict = verdict_agent.run_sync(
                        user_prompt=final_question,
                        message_history=combined_messages_for_verdict
                    )
                    
                    # Extract JSON from response
                    raw_response = result_verdict.data
                    with st.expander("Show raw response"):
                        st.markdown(raw_response)
                    
                    # Parse JSON response
                    json_data = extract_json_from_text(raw_response)
                    
                    if json_data:
                        # Validate using Pydantic
                        try:
                            verdict = CaseVerdict(**json_data)
                            
                            # Create a formatted display of the verdict
                            verdict_color = "red" if verdict.verdict == "Guilty" or verdict.verdict == "For Plaintiff" else "green"
                            
                            st.markdown(f"""
                            ### Verdict: <span style='color:{verdict_color}'>{verdict.verdict}</span>
                            
                            **Confidence Score**: {verdict.confidence_score}%
                            
                            **Reasoning**:
                            {verdict.reasoning}
                            
                            **Sentence/Judgment Recommendation**:
                            {verdict.sentence_recommendation}
                            
                            **Appeal Potential**:
                            {verdict.appeal_potential}
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error validating response format: {e}")
                            st.markdown("### Verdict Summary")
                            st.markdown(raw_response)
                    else:
                        st.error("Could not parse a structured verdict. Showing raw response:")
                        st.markdown(raw_response)
                except Exception as e:
                    st.error(f"❌ Error generating verdict: {str(e)}")
                    st.markdown("**Verdict Unavailable**")
                    st.markdown("Due to a technical error, a formal verdict could not be generated for this case.")
                
            # Add a disclaimer
            st.markdown("---")
            st.caption("**Disclaimer**: This tool provides automated legal analysis for educational purposes only. It does not constitute legal advice. Always consult with a qualified attorney for legal matters.")

# Run analysis button
if st.button("Analyze Case", type="primary", use_container_width=True):
    run_case_analysis()
else:
    # Initial page content
    st.info("👈 Configure your case parameters in the sidebar and click 'Analyze Case' to get started.")
    
    st.markdown(f"""
    ### How this works
    1. Enter your case details in the sidebar
    2. Click the "Test Connections" button to verify your API connections work correctly
    3. Set your case parameters and click 'Analyze Case' to start the analysis
    4. The app will generate both defense arguments and prosecution arguments
    5. A final verdict will be provided based on all arguments presented
    
    This tool uses multiple AI models to simulate a legal process, providing structured analysis of legal cases.
    """)
