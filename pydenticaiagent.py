import os
import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
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

# Custom CSS for theming (since the theme parameter in set_page_config is experimental)
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

# OpenAI API Key
openai_api_key=os.getenv('OPENAI_API_KEY')
with st.sidebar.expander("Local LLM Setting"):
    use_ollama = st.checkbox("Use Ollama for reasoning", value=True)
    if use_ollama:
        ollama_url = st.text_input("Ollama Base URL", 
                                   value=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1'))
        ollama_model = st.text_input("Ollama Model", value="deepseek-r1")

# Function to strip thinking tags
def strip_thinking_tags(response):
    """Remove <think>...</think> tags from model response"""
    cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    return cleaned_response.strip() if cleaned_response.strip() else response

# Format the user profile and constraints based on inputs
user_profile = f"a {age} year old {profession} with ${net_worth/1000000:.1f}M net worth and {investment_window} year {risk_tolerance.lower()} investment window"
investment_constraint = f"never invest more than {max_allocation}% of their portfolio into any individual stock or ETF"
allocation_range = f"{min_allocation}-{max_allocation}%"

# Main analysis function
def run_investment_analysis():
    with st.spinner("Initializing AI models..."):
        try:
            # Primary model for generating arguments (using OpenAI)
            generation_model = OpenAIModel(
                model_name='gpt-4o-mini',
                provider=OpenAIProvider(api_key=openai_api_key)
            )

            # Secondary model for reasoning/synthesis
            if use_ollama:
                reasoning_model = OpenAIModel(
                    model_name=ollama_model,
                    provider=OpenAIProvider(
                        api_key=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1'),
                        base_url=ollama_url
                    )
                )
            else:
                # Fallback to using OpenAI for reasoning as well
                reasoning_model = generation_model
                
            st.success("Models initialized successfully")
        except Exception as e:
            st.error(f"Error initializing AI models: {e}")
            return

    # System prompts
    generation_system_prompt = f"""
    You are an investment portfolio manager specialized in US tech stocks.
    The user is {user_profile}.
    Your job is to provide investment advice on the {stock_name} stock.
    The user will {investment_constraint}.
    Your task is to analyze the stock based on the user's request.
    If suggesting an investment, recommend an allocation between {allocation_range}.
    Consider the user's profile, investment window, the stock's performance, market trends, and expert opinions.
    Briefly mention potential alternative investments if relevant to the user's profile and goals.
    Be clear and concise.
    """

    reasoning_system_prompt = f"""
    You are an AI assistant synthesizing investment arguments.
    You have been provided with a conversation containing arguments for and against investing in {stock_name} for {user_profile}.
    Based *only* on the provided conversation history, provide a final summary and recommendation.
    Answer the user's final question: "Should I buy {stock_name} stock?".
    If recommending to buy, suggest an allocation percentage within the user's {allocation_range} constraint, justifying it based on the provided arguments.
    Acknowledge the user's {risk_tolerance.lower()} {investment_window}-year window.
    """

    # Agent definitions
    generation_agent = Agent(model=generation_model, system_prompt=generation_system_prompt)
    reasoning_agent = Agent(model=reasoning_model, system_prompt=reasoning_system_prompt)

    # Create tabs for the different analysis sections
    tab1, tab2, tab3 = st.tabs(["Pro Arguments", "Counter Arguments", "Final Recommendation"])

    # 1. Get arguments FOR buying the stock
    with tab1:
        with st.spinner("Generating pro-investment arguments..."):
            st.subheader(f"Arguments FOR investing in {stock_name}")
            pro_prompt = f"Provide arguments why I should buy {stock_name} stock and how much I should invest (as a percentage between {allocation_range})."
            
            result_pro = generation_agent.run_sync(user_prompt=pro_prompt)
            st.markdown(result_pro.data)
            
            # Store history
            history_after_pro = result_pro.all_messages()
            pro_new_messages = result_pro.new_messages()

    # 2. Get arguments AGAINST buying the stock
    with tab2:
        with st.spinner("Generating counter-arguments..."):
            st.subheader(f"Arguments AGAINST investing in {stock_name}")
            con_prompt = f"Now, provide counter-arguments why I should *not* buy {stock_name} stock and what alternative actions or investments I could consider instead."
            
            result_con = generation_agent.run_sync(
                user_prompt=con_prompt,
                message_history=history_after_pro
            )
            st.markdown(result_con.data)
            
            # Store messages
            con_new_messages = result_con.new_messages()

    # 3. Combine for reasoning
    combined_messages_for_reasoning = pro_new_messages + con_new_messages

    # 4. Final recommendation
    with tab3:
        with st.spinner("Synthesizing final recommendation..."):
            st.subheader("Final Investment Recommendation")
            final_question = f"Based on the arguments presented, should I buy {stock_name} stock?"
            
            result_reasoning = reasoning_agent.run_sync(
                user_prompt=final_question,
                message_history=combined_messages_for_reasoning
            )
            # Clean the response to remove thinking tags
            #cleaned_response = strip_thinking_tags(result_reasoning.data)
            # Implement  a Pydantic model for structured final output on the reasoning model
            # Display the cleaned final result in a highlighted box
            #st.info(cleaned_response)
            st.info(result_reasoning.data)
            
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
    3. The app will generate both pro and counter arguments for investing in {stock_name}
    4. A final recommendation will be provided based on your specific profile
    
    This tool uses AI to simulate the advice of an investment portfolio manager, providing a balanced view of potential investments.
    """)