import streamlit as st
from workflow import build_agentic_workflow

# Configure the Streamlit App
st.set_page_config(
    page_title="Agentic AI Investment Advisor", 
    page_icon="📈", 
    layout="wide"
)

st.markdown("<h1 style='color:blue;'>Agentic AI Investment Advisor by Santosh Jadhav</h1>", unsafe_allow_html=True)

# Define Sidebar Inputs
with st.sidebar:
    st.header("Investor Profile")
    
    income = st.number_input(
        "Investment Income (₹)", 
        min_value=100000, 
        max_value=10000000, 
        value=4500000, 
        step=500000,
        help="Annual income in INR"
    )
    
    age = st.number_input(
        "Age", 
        min_value=18, 
        max_value=100, 
        value=35, 
        step=1
    )
    
    risk_appetite = st.selectbox(
        "Risk Appetite", 
        ["Low", "Moderate", "High", "Speculative"], 
        index=2
    )
    
    loss_tolerance = st.selectbox(
        "Loss Tolerance", 
        ["Low", "Moderate", "High"], 
        index=1
    )

	# Custom CSS to make the button blue
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #0066cc;
        color: white;
        border-color: #0066cc;
    }
    div.stButton > button:first-child:hover {
        background-color: #0052a3;
        color: white;
        border-color: #0052a3;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    generate_btn = st.button("Generate Strategy", use_container_width=True)

# Main Application Logic
if generate_btn:
    st.header("Investment Advisory – Report")
    
    # Container for progress and output
    with st.spinner("Agents are analyzing market data and compiling your custom report..."):
        try:
            # 1. Initialize workflow
            app_workflow = build_agentic_workflow()
            
            # 2. Package inputs
            user_input = {
                "income": float(income),
                "age": int(age),
                "risk_appetite": risk_appetite,
                "loss_tolerance": loss_tolerance
            }
            
            # 3. Invoke LangGraph
            result = app_workflow.invoke(user_input)
            
            # 4. Display result
            st.success("Analysis Complete!")
            st.markdown(result["final_recommendation"])
            
        except Exception as e:
            st.error(f"An error occurred during workflow execution: {str(e)}")
            st.info("Ensure all your API keys (Groq, OpenAI, Serper) are correctly set in the .env file.")
else:
    # Default landing screen
    st.info("👈 Please enter your financial profile in the sidebar and click **Generate Strategy** to receive your AI-tailored investment report.")

# --- ADDED FOOTER SECTION ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: small;'>
        <p style='margin-bottom: 2px;'>Prepared by: AI Agent as Senior Investment Advisor</p>
        <p style='margin-top: 0px;'>Disclaimer: This report is for informational purposes only and does not constitute an offer or solicitation to buy or sell any securities. Investment decisions should be made in consultation with your personal financial advisor, taking into account your individual circumstances, risk tolerance, and investment horizon.</p>
    </div>
    """, 
    unsafe_allow_html=True
)