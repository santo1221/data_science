import json
import os
from typing import TypedDict, Dict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools.google_serper import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper

# Load environment variables
load_dotenv()

# Define State
class InvestorState(TypedDict):
    income: float
    age: int
    risk_appetite: str
    loss_tolerance: str
    asset_allocation: Dict[str, str]
    market_research: str
    final_recommendation: str

# Initialize LLM and Tools
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1) 
yfinance_tool = YahooFinanceNewsTool()

api_wrapper = GoogleSerperAPIWrapper(serper_api_key=os.environ.get("SERPER_API_KEY"), type="news")
serper_search = GoogleSerperRun(
    api_wrapper=api_wrapper, 
    handle_tool_error="System Error: The Google Serper API is currently unreachable or the query was invalid."
)

# --- 1. Profiler Agent ---
def profiler_agent(state: InvestorState) -> InvestorState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Determine Indian asset allocation (JSON) for: Equity_Stocks, Equity_MF, Debt, Gold."),
        ("user", "Age: {age}, Risk: {risk_appetite}, Income: {income}, Loss_tolerance: {loss_tolerance}")
    ])
    chain = prompt | llm
    response = chain.invoke({"age": state["age"], "risk_appetite": state["risk_appetite"], "income": state["income"], "loss_tolerance": state["loss_tolerance"]})
    
    # Basic cleaning of LLM output
    content = response.content.replace("```json", "").replace("```", "").strip()
    try:
        allocation = json.loads(content)
    except json.JSONDecodeError:
        allocation = {"Fallback": "Failed to parse JSON, check LLM output."}
        
    return {"asset_allocation": allocation}

# --- 2. Researcher Agent ---
def researcher_agent(state: InvestorState) -> InvestorState:
    risk = state["risk_appetite"]
    
    # Dynamically determine tickers using an LLM call
    ticker_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Indian stock market quantitative analyst. 
        Based on the user's risk appetite, output a comma-separated list of 3-4 highly relevant NSE stock ticker symbols. 
        You MUST append '.NS' to every ticker (e.g., RELIANCE.NS, TCS.NS). 
        - For High risk/Speculative: Choose fundamentally strong Small-Cap stocks.
        - For Moderate risk: Choose high-growth Mid-Cap stocks.
        - For Low risk: Choose stable Large-Cap Blue-Chips.
        ONLY output the comma-separated tickers, nothing else."""),
        ("user", "Risk Appetite: {risk_appetite}")
    ])
    
    ticker_chain = ticker_prompt | llm
    ticker_response = ticker_chain.invoke({"risk_appetite": risk})
    dynamic_tickers = ticker_response.content.strip().replace("`", "")
    
    ticker_list = [t.strip() for t in dynamic_tickers.split(",") if t.strip()]
    
    news_data_list = []
    
    for ticker in ticker_list:
        ticker_news = ""
        yfinance_success = False
        
        # PRIMARY ATTEMPT: Yahoo Finance
        try:
            ticker_news = yfinance_tool.run(ticker)
            if ticker_news and not any(err in ticker_news.lower() for err in ["error", "not found", "no news found"]):
                yfinance_success = True
                news_data_list.append(f"--- News for {ticker} (via Yahoo Finance) ---\n{ticker_news}")
        except Exception:
            pass # yfinance_success remains False
            
        # SECONDARY ATTEMPT: Google Serper Fallback
        if not yfinance_success:
            clean_ticker = ticker.replace(".NS", "")
            search_query = f"Latest financial news and market sentiment for Indian stock: {clean_ticker}"
            try:
                fallback_news = serper_search.run(search_query)
                news_data_list.append(f"--- News for {clean_ticker} (via Web Search) ---\n{fallback_news}")
            except Exception:
                continue

    news_data = "\n\n".join(news_data_list)
    if not news_data.strip():
        news_data = "Could not retrieve live data for any tickers. Rely on general market knowledge."

    # Summarize the Research
    research_summarizer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial researcher. Summarize the following news specifically for an Indian investor. 
        Extract the core sentiment, highlight potential stock picks from the provided tickers, and identify any immediate macroeconomic risks.
        If the news data mentions errors, provide the best analysis you can based on your inherent knowledge."""),
        ("user", "Tickers Researched: {tickers}\n\nNews Data:\n{news}")
    ])
    
    summary_chain = research_summarizer_prompt | llm
    summary = summary_chain.invoke({
        "tickers": ", ".join(ticker_list),
        "news": news_data
    })
    
    return {"market_research": summary.content}

# --- 3. Compiler / Investment Advisor Agent ---
def compiler_agent(state: InvestorState) -> InvestorState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Investment Advisor. Create a final report using the provided allocation and the live market research summary."),
        ("user", "Allocation: {allocation}\n\nLive Research: {research}")
    ])
    chain = prompt | llm
    response = chain.invoke({"allocation": state["asset_allocation"], "research": state["market_research"]})
    return {"final_recommendation": response.content}

# --- Graph Orchestration ---
def build_agentic_workflow():
    workflow = StateGraph(InvestorState)

    workflow.add_node("Profiler_Agent", profiler_agent)
    workflow.add_node("Researcher_Agent", researcher_agent)
    workflow.add_node("Investment_Advisor_Agent", compiler_agent)

    workflow.set_entry_point("Profiler_Agent")
    workflow.add_edge("Profiler_Agent", "Researcher_Agent")
    workflow.add_edge("Researcher_Agent", "Investment_Advisor_Agent")
    workflow.add_edge("Investment_Advisor_Agent", END)

    return workflow.compile()
