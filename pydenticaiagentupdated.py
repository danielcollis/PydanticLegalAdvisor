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
    page_title="Investment Advisor",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for theming
st.markdown("""
<style>
    div.stButton > button[kind="primary"] {
        background-color: #E48908FF;
        color: white;
    }
    div.stSlider > div[data-baseweb="slider"] > div > div {
        background-color: #F09B12FF !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        border-color: #041421FF;
    }
    h1, h2, h3 {
        color: #0D47A1;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f5;
    }
</style>
""", unsafe_allow_html=True)

# Role assignment for different LLMs:

# Deepseek locally (via Ollama) now focuses on generating pro-investment arguments
# GPT-4o-mini generates counter-arguments
# GPT-4o makes the final decision

# App title and description
st.title("AI-Powered Investment Advisor")
st.markdown("### Smart investment analysis using AI Agent")

# Sidebar for user inputs
st.sidebar.header("Investment Parameters")

# Popular stocks list for dropdown
popular_stocks = [
    "Apple (AAPL)", 
    "Microsoft (MSFT)", 
    "Amazon (AMZN)", 
    "Alphabet (GOOGL)", 
    "NVIDIA (NVDA)", 
    "Tesla (TSLA)", 
    "Meta (META)", 
    "Berkshire Hathaway (BRK.B)", 
    "JPMorgan Chase (JPM)", 
    "Johnson & Johnson (JNJ)"
]

# Stock selection dropdown
selected_stock = st.sidebar.selectbox(
    "Select Stock", 
    options=popular_stocks
)

# Extract just the stock name from the selection (removing ticker)
# It uses the string " (" (a space followed by an opening parenthesis) as the delimiter at which to split the original string
stock_name = selected_stock.split(" (")[0]

st.sidebar.subheader("Investor Profile")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=45)
profession = st.sidebar.text_input("Profession", value="tech executive")
net_worth = st.sidebar.number_input("Net Worth (in USD)", 
                                   min_value=10000, 
                                   max_value=1000000000, 
                                   value=10000000,
                                   format="%d")
investment_window = st.sidebar.slider("Investment Window (years)", 1, 30, 5)
risk_tolerance = st.sidebar.select_slider(
    "Risk Tolerance",
    options=["Conservative", "Moderate", "Aggressive"],
    value="Aggressive"
)

st.sidebar.subheader("Investment Constraints")
max_allocation = st.sidebar.slider("Maximum Investment (%)", 1, 20, 5,
                                  help="Maximum percentage of portfolio to invest in a single stock")
min_allocation = st.sidebar.slider("Minimum Investment (%)", 0, 10, 1,
                                  help="Minimum percentage of portfolio to invest in a single stock")

# API Keys
openai_api_key = os.getenv('OPENAI_API_KEY')
with st.sidebar.expander("LLM Settings"):
    ollama_url = st.text_input("Ollama Base URL", 
                               value=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1'))
    ollama_model = st.text_input("Ollama Model", value="deepseek-r1")
    gpt4o_mini_model = st.text_input("GPT-4o Mini Model", value="gpt-4o-mini")
    gpt4o_model = st.text_input("GPT-4o Model", value="gpt-4o")

# Pydantic model for structured output
class InvestmentRecommendation(BaseModel):
    should_invest: bool = Field(description="Whether the user should invest in the stock")
    recommended_allocation_percentage: float = Field(description="Recommended allocation percentage (between min and max constraints)")
    reasoning: str = Field(description="Explanation for the investment recommendation")
    alternative_investments: str = Field(description="Suggested alternative investments if applicable")

# Format the user profile 
user_profile = f"a {age} year old {profession} with ${net_worth/1000000:.1f}M net worth and {investment_window} year {risk_tolerance.lower()} investment window"
investment_constraint = f"never invest more than {max_allocation}% of their portfolio into any individual stock or ETF"
allocation_range = f"{min_allocation}-{max_allocation}%"

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
def run_investment_analysis():
    with st.spinner("Initializing AI models..."):
        try:
            # Pro arguments model (using deepseek via Ollama)
            pro_model = OpenAIModel(
                model_name=ollama_model,
                provider=OpenAIProvider(
                    api_key=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1'),
                    base_url=ollama_url
                )
            )
            
            # Con arguments model (using GPT-4o-mini)
            con_model = OpenAIModel(
                model_name=gpt4o_mini_model,
                provider=OpenAIProvider(api_key=openai_api_key)
            )
            
            # Final decision model (using GPT-4o)
            decision_model = OpenAIModel(
                model_name=gpt4o_mini_model,
                provider=OpenAIProvider(api_key=openai_api_key)
            )
                
            st.success("Models initialized successfully")
        except Exception as e:
            st.error(f"Error initializing AI models: {e}")
            return

    # System prompts
    pro_system_prompt = f"""
    You are an investment portfolio manager specialized in US tech stocks with a bullish mindset.
    The user is {user_profile}.
    Your job is to provide ONLY positive investment advice on the {stock_name} stock.
    Focus on all positive aspects - strong growth potential, company performance, future prospects, and market advantages.
    If suggesting an investment, recommend an allocation between {allocation_range}.
    Be optimistic and persuasive but realistic.
    """

    con_system_prompt = f"""
    You are an investment portfolio manager specialized in US tech stocks with a cautious and skeptical mindset.
    The user is {user_profile}.
    Your job is to provide ONLY critical investment advice against investing in {stock_name} stock.
    Focus on all negative aspects - risks, downsides, market threats, overvaluation concerns, and potential problems.
    Suggest alternative investments that might be safer or have better returns.
    Be critical and cautious but factual.
    """

    decision_system_prompt = f"""
    You are a balanced and objective investment advisor making the final recommendation.
    The user is {user_profile}.
    You have been provided with both bullish and bearish arguments about {stock_name}.
    Using ONLY the provided arguments, make a final recommendation on whether the user should invest in {stock_name}.
    If recommending to invest, suggest an allocation percentage within the user's {allocation_range} constraint.
    Always consider the user's {risk_tolerance.lower()} risk tolerance and {investment_window}-year investment window.
    
    You MUST structure your response as a valid JSON object with the following fields:
    - should_invest (boolean): Whether the user should invest in the stock
    - recommended_allocation_percentage (number): Recommended allocation percentage between {min_allocation} and {max_allocation}
    - reasoning (string): Explanation for the investment recommendation
    - alternative_investments (string): Suggested alternative investments if applicable
    
    Format your JSON response within ```json ``` code blocks.
    """

    # Agent definitions
    pro_agent = Agent(model=pro_model, system_prompt=pro_system_prompt)
    con_agent = Agent(model=con_model, system_prompt=con_system_prompt)
    decision_agent = Agent(model=decision_model, system_prompt=decision_system_prompt)

    # Create tabs for the different analysis sections
    tab1, tab2, tab3 = st.tabs(["Pro Arguments", "Counter Arguments", "Final Recommendation"])

    # 1. Get arguments FOR buying the stock
    with tab1:
        with st.spinner("Generating pro-investment arguments..."):
            st.subheader(f"Arguments FOR investing in {stock_name}")
            pro_prompt = f"Provide strong arguments why I should buy {stock_name} stock and how much I should invest (as a percentage between {allocation_range})."
            
            result_pro = pro_agent.run_sync(user_prompt=pro_prompt)
            st.markdown(result_pro.data)
            
            # Store history
            history_after_pro = result_pro.all_messages()
            pro_new_messages = result_pro.new_messages()

    # 2. Get arguments AGAINST buying the stock
    with tab2:
        with st.spinner("Generating counter-arguments..."):
            st.subheader(f"Arguments AGAINST investing in {stock_name}")
            con_prompt = f"Provide strong counter-arguments why I should *not* buy {stock_name} stock and what alternative actions or investments I could consider instead."
            
            result_con = con_agent.run_sync(user_prompt=con_prompt)
            st.markdown(result_con.data)
            
            # Store messages
            con_new_messages = result_con.new_messages()

    # 3. Combine for decision making
    combined_messages_for_decision = pro_new_messages + con_new_messages

    # 4. Final recommendation
    with tab3:
        with st.spinner("Synthesizing final recommendation..."):
            st.subheader("Final Investment Recommendation")
            final_question = f"""Based on the arguments presented, should I buy {stock_name} stock? 

Please provide your recommendation in JSON format with these fields:
- should_invest (boolean)
- recommended_allocation_percentage (number between {min_allocation} and {max_allocation})
- reasoning (string)
- alternative_investments (string)"""
            
            result_decision = decision_agent.run_sync(
                user_prompt=final_question,
                message_history=combined_messages_for_decision
            )
            
            # Extract JSON from response
            raw_response = result_decision.data
            st.text("Model Response:")
            with st.expander("Show raw response"):
                st.markdown(raw_response)
            
            # Parse JSON response
            json_data = extract_json_from_text(raw_response)
            
            if json_data:
                # Validate using Pydantic
                try:
                    recommendation = InvestmentRecommendation(**json_data)
                    
                    # Create a formatted display of the recommendation
                    decision_color = "green" if recommendation.should_invest else "red"
                    decision_text = "INVEST" if recommendation.should_invest else "DO NOT INVEST"
                    
                    st.markdown(f"""
                    ### Decision: <span style='color:{decision_color}'>{decision_text}</span>
                    
                    **Recommended Allocation**: {recommendation.recommended_allocation_percentage}%
                    
                    **Reasoning**:
                    {recommendation.reasoning}
                    
                    **Alternative Investments to Consider**:
                    {recommendation.alternative_investments}
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error validating response format: {e}")
                    st.markdown("### Decision Summary")
                    st.markdown(raw_response)
            else:
                st.error("Could not parse a structured recommendation. Showing raw response:")
                st.markdown(raw_response)
            
            # Add a disclaimer
            st.markdown("---")
            st.caption("**Disclaimer**: This tool provides automated investment analysis for educational purposes only. Always consult with a professional financial advisor before making investment decisions.")

# Run analysis button
if st.button("Analyze Investment", type="primary", use_container_width=True):
    run_investment_analysis()
else:
    # Initial page content
    st.info("👈 Configure your investment parameters in the sidebar and click 'Analyze Investment' to get started.")
    
    st.markdown(f"""
    ### How this works
    1. Select your stock from the dropdown and configure your investor profile in the sidebar
    2. Click 'Analyze Investment' to start the analysis
    3. The app will generate both pro arguments (using Deepseek) and counter arguments (using GPT-4o-mini)
    4. A final structured recommendation will be provided by GPT-4o based on your specific profile
    
    This tool uses multiple AI models to simulate a balanced investment advisory process, providing structured recommendations tailored to your profile.
    """)