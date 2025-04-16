import os
import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field
import json
import re

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
        background-color: #2E4053;
        color: white;
    }
    div.stSlider > div[data-baseweb="slider"] > div > div {
        background-color: #34495E !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        border-color: #2E4053;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f5;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("AI-Powered Legal Case Analyzer")
st.markdown("### Multi-agent legal case analysis and outcome prediction")

# Sidebar for user inputs
st.sidebar.header("Case Parameters")

# Case type dropdown
case_types = [
    "Criminal - Theft",
    "Criminal - Assault",
    "Criminal - Fraud",
    "Civil - Contract Dispute",
    "Civil - Personal Injury",
    "Civil - Property Dispute",
    "Family - Divorce",
    "Family - Child Custody",
    "Intellectual Property - Copyright",
    "Intellectual Property - Patent"
]

case_type = st.sidebar.selectbox(
    "Case Type", 
    options=case_types
)

st.sidebar.subheader("Case Details")
defendant_name = st.sidebar.text_input("Defendant Name", value="John Smith")
plaintiff_name = st.sidebar.text_input("Plaintiff/Prosecution Name", value="State of California")
jurisdiction = st.sidebar.selectbox(
    "Jurisdiction",
    options=["Federal", "State", "Local"]
)

# Case facts text area
case_facts = st.sidebar.text_area(
    "Case Facts", 
    height=200,
    value="""The defendant is accused of taking $5,000 from their employer's cash register over a period of three months. 
Security footage shows the defendant accessing the register after hours on multiple occasions. 
The defendant claims they were authorized to make change for the next day."""
)

# Previous convictions/history
previous_history = st.sidebar.text_area(
    "Previous Legal History", 
    height=100,
    value="No prior convictions. Has worked at the company for 5 years with positive performance reviews."
)

# Evidence strength slider
evidence_strength = st.sidebar.slider(
    "Evidence Strength", 
    1, 10, 7,
    help="How strong is the evidence against the defendant (1=weak, 10=strong)"
)

# API Keys
openai_api_key = os.getenv('OPENAI_API_KEY')
with st.sidebar.expander("LLM Settings"):
    ollama_url = st.text_input("Ollama Base URL", 
                               value=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1'))
    ollama_model = st.text_input("Ollama Model", value="deepseek-r1")
    gpt4o_mini_model = st.text_input("GPT-4o Mini Model", value="gpt-4o-mini")
    gpt4o_model = st.text_input("GPT-4o Model", value="gpt-4o")

# Pydantic model for structured output
class CaseVerdict(BaseModel):
    guilty_or_liable: bool = Field(description="Whether the defendant is found guilty (criminal) or liable (civil)")
    confidence_level: int = Field(description="Confidence level in the verdict (1-10)")
    reasoning: str = Field(description="Explanation for the verdict decision")
    sentencing_recommendation: str = Field(description="Recommended sentence or damages if applicable")

# Format the case summary
case_summary = f"Case Type: {case_type}\nDefendant: {defendant_name}\nPlaintiff/Prosecution: {plaintiff_name}\nJurisdiction: {jurisdiction}"

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

# Main analysis function
def run_case_analysis():
    with st.spinner("Initializing AI models..."):
        try:
            # Defense model (using deepseek via Ollama)
            defense_model = OpenAIModel(
                model_name=ollama_model,
                provider=OpenAIProvider(
                    base_url=ollama_url
                )
            )
            
            # Prosecution model (using GPT-4o-mini)
            prosecution_model = OpenAIModel(
                model_name=gpt4o_mini_model,
                provider=OpenAIProvider(api_key=openai_api_key)
            )
            
            # Judge/Jury model (using GPT-4o)
            judge_model = OpenAIModel(
                model_name=gpt4o_model,
                provider=OpenAIProvider(api_key=openai_api_key)
            )
                
            st.success("Models initialized successfully")
        except Exception as e:
            st.error(f"Error initializing AI models: {e}")
            return

    # System prompts
    defense_system_prompt = f"""
    You are an experienced defense attorney tasked with defending {defendant_name} in a {case_type} case.
    
    Case Summary:
    {case_summary}
    
    Case Facts:
    {case_facts}
    
    Defendant's Previous History:
    {previous_history}
    
    Your job is to build the strongest possible defense for your client. 
    Consider all possible legal defenses, mitigating factors, procedural issues, and reasonable doubt arguments.
    Be thorough, persuasive, and focus on evidence interpretation and legal precedents that support your client.
    Structure your response with clear arguments, citing relevant legal principles when applicable.
    """

    prosecution_system_prompt = f"""
    You are an experienced prosecutor (in criminal cases) or plaintiff's attorney (in civil cases) arguing against {defendant_name} in a {case_type} case.
    
    Case Summary:
    {case_summary}
    
    Case Facts:
    {case_facts}
    
    Defendant's Previous History:
    {previous_history}
    
    Evidence Strength: {evidence_strength}/10
    
    Your job is to build the strongest possible case against the defendant.
    Identify all weaknesses in the defense's arguments, highlight incriminating evidence, and establish the elements of the offense.
    Focus on legal standards of proof, relevant statutes, and precedents that support finding the defendant guilty or liable.
    Structure your response with clear arguments, citing relevant legal principles when applicable.
    """

    judge_system_prompt = f"""
    You are an impartial judge or jury determining the outcome of a {case_type} case involving {defendant_name}.
    
    Case Summary:
    {case_summary}
    
    Case Facts:
    {case_facts}
    
    Defendant's Previous History:
    {previous_history}
    
    Evidence Strength: {evidence_strength}/10
    
    You have been provided with both defense and prosecution arguments.
    Using ONLY the provided arguments and case facts, determine an outcome for this case.
    Consider the applicable standard of proof (beyond reasonable doubt for criminal cases, preponderance of evidence for civil cases).
    
    You MUST structure your response as a valid JSON object with the following fields:
    - guilty_or_liable (boolean): Whether the defendant is found guilty (criminal) or liable (civil)
    - confidence_level (number): Confidence level in the verdict (1-10)
    - reasoning (string): Explanation for the verdict decision
    - sentencing_recommendation (string): Recommended sentence or damages if applicable
    
    Format your JSON response within ```json ``` code blocks.
    """

    # Agent definitions
    defense_agent = Agent(model=defense_model, system_prompt=defense_system_prompt)
    prosecution_agent = Agent(model=prosecution_model, system_prompt=prosecution_system_prompt)
    judge_agent = Agent(model=judge_model, system_prompt=judge_system_prompt)

    # Create tabs for the different analysis sections
    tab1, tab2, tab3 = st.tabs(["Defense Arguments", "Prosecution Arguments", "Verdict"])

    # 1. Get defense arguments
    with tab1:
        with st.spinner("Generating defense arguments..."):
            st.subheader("Defense Arguments")
            defense_prompt = f"Please provide a comprehensive defense for {defendant_name} in this {case_type} case."
            
            result_defense = defense_agent.run_sync(user_prompt=defense_prompt)
            st.markdown(result_defense.data)
            
            # Store history
            defense_new_messages = result_defense.new_messages()

    # 2. Get prosecution arguments
    with tab2:
        with st.spinner("Generating prosecution arguments..."):
            st.subheader("Prosecution Arguments")
            prosecution_prompt = f"Please provide a comprehensive prosecution case against {defendant_name} in this {case_type} case."
            
            result_prosecution = prosecution_agent.run_sync(user_prompt=prosecution_prompt)
            st.markdown(result_prosecution.data)
            
            # Store messages
            prosecution_new_messages = result_prosecution.new_messages()

    # 3. Combine for verdict
    combined_messages_for_verdict = defense_new_messages + prosecution_new_messages

    # 4. Final verdict
    with tab3:
        with st.spinner("Deliberating on final verdict..."):
            st.subheader("Case Verdict")
            verdict_prompt = f"""Based on the defense and prosecution arguments presented, determine the verdict for {defendant_name} in this {case_type} case.

Please provide your verdict in JSON format with these fields:
- guilty_or_liable (boolean)
- confidence_level (number between 1 and 10)
- reasoning (string)
- sentencing_recommendation (string)"""
            
            result_verdict = judge_agent.run_sync(
                user_prompt=verdict_prompt,
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
                    verdict_result = "GUILTY" if verdict.guilty_or_liable else "NOT GUILTY"
                    if "Civil" in case_type:
                        verdict_result = "LIABLE" if verdict.guilty_or_liable else "NOT LIABLE"
                    
                    verdict_color = "red" if verdict.guilty_or_liable else "green"
                    
                    st.markdown(f"""
                    ## Verdict: <span style='color:{verdict_color}'>{verdict_result}</span>
                    
                    **Confidence Level**: {verdict.confidence_level}/10
                    
                    **Reasoning**:
                    {verdict.reasoning}
                    
                    **Sentencing/Damages Recommendation**:
                    {verdict.sentencing_recommendation}
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error validating response format: {e}")
                    st.markdown("### Verdict Summary")
                    st.markdown(raw_response)
            else:
                st.error("Could not parse a structured verdict. Showing raw response:")
                st.markdown(raw_response)
            
            # Add a disclaimer
            st.markdown("---")
            st.caption("**Disclaimer**: This tool provides automated legal analysis for educational purposes only. It is not a substitute for professional legal advice.")

# Run analysis button
if st.button("Analyze Case", type="primary", use_container_width=True):
    run_case_analysis()
else:
    # Initial page content
    st.info("👈 Enter your case details in the sidebar and click 'Analyze Case' to get started.")
    
    st.markdown(f"""
    ### How this works
    1. Enter the case type, defendant and plaintiff information, and jurisdiction in the sidebar
    2. Provide case facts and any relevant previous history
    3. Click 'Analyze Case' to start the analysis
    4. The app will generate defense arguments (using Deepseek) and prosecution arguments (using GPT-4o-mini)
    5. A final verdict will be determined by GPT-4o based on the arguments presented
    
    This tool uses multiple AI models to simulate a legal proceeding, providing structured analysis of the case facts and a predicted outcome.
    """)