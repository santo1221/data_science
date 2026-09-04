import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from a .env file if present
load_dotenv()

# Import storage and retriever modules with fallbacks
try:
    from langchain.storage import LocalFileStore, create_kv_docstore
    from langchain.retrievers.multi_vector import MultiVectorRetriever
except ImportError:
    from langchain_classic.storage import LocalFileStore, create_kv_docstore
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# =========================================================
# 1. Page Configuration & UI Layout
# =========================================================
st.set_page_config(
    page_title="IS 800:2007 Structural Engineering Assistant",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ IS 800:2007 Steel Design Assistant")
st.subheader("Developed by Smita Jadhav")
st.caption("Ask technical questions regarding structural engineering design, calculations, and IS 800:2007 compliance.")

# Ensure keys are present in the environment before proceeding
if not os.environ.get("OPENROUTER_API_KEY") or not os.environ.get("PINECONE_API_KEY"):
    st.error("⚠️ API keys are missing! Please set OPENROUTER_API_KEY and PINECONE_API_KEY as environment variables or in your .env file.")
    st.stop()

# =========================================================
# 2. Cached Resource Initialization
# =========================================================
@st.cache_resource
def load_rag_retriever():
    """
    Loads and caches the heavy vectorstore, embedding model, 
    and local docstore connection to prevent reloading on every rerun.
    """
    parent_store_dir = "./RAG_Civil_IS_Codes/parent_docs"
    
    if not os.path.exists(parent_store_dir):
        st.error(f"Directory '{parent_store_dir}' not found. Please ensure parent documents exist.")
        st.stop()

    # Connect to Parent Docstore (Local Disk)
    file_store = LocalFileStore(parent_store_dir)
    parent_docstore = create_kv_docstore(file_store)

    # Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Connect to Pinecone Index
    index_name = "civil-is-child"
    child_vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    # Create MultiVectorRetriever with MMR
    retriever = MultiVectorRetriever(
        vectorstore=child_vectorstore,
        docstore=parent_docstore,
        id_key="parent_id",
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.95
        }
    )
    
    return retriever


try:
    mmr_retriever = load_rag_retriever()
except Exception as e:
    st.error(f"Failed to load RAG components: {e}")
    st.stop()

# Initialize LLM via OpenRouter
llm = ChatOpenAI(
    model="qwen/qwen3-30b-a3b",
    temperature=0.1,
    streaming=True,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# =========================================================
# 3. System Prompt & RAG Chain Construction
# =========================================================
system_prompt = (
    "You are a Civil Structural Engineering Expert specializing in steel design as per IS 800:2007 (General Construction in Steel - Code of Practice).\n"
    "Your role is to provide accurate, concise, and professional responses to queries related to structural engineering design, analysis, and detailing, strictly aligned with IS 800:2007 provisions.\n\n"
    "Guidelines:\n"
    "- Always ground your answers in IS 800:2007 clauses, tables, and design provisions.\n"
    "- When citing, mention the relevant clause number or section (e.g., 'as per IS 800:2007, Clause 5.4.1').\n"
    "- Provide step-by-step explanations for calculations, design checks, and code compliance.\n"
    "- Use clear technical language suitable for practicing engineers, project managers, and students.\n"
    "- Do not provide the full copyrighted text of IS 800:2007; instead, summarize and reference clauses.\n"
    "- Where applicable, illustrate with formulas, design examples, or practical applications.\n"
    "- If a query is outside the scope of IS 800:2007, clarify the limitation and suggest the relevant standard (e.g., IS 456 for concrete, IS 1893 for seismic design).\n"
    "- Maintain a professional, authoritative, and engineering-focused tone.\n\n"
    "Objective:\n"
    "Deliver reliable, code-compliant guidance that helps engineers design safe, efficient, and practical steel structures in accordance with IS 800:2007.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL Chain incorporating context retrieval and chat history
rag_chain = (
    {
        "context": (lambda x: x["question"]) | mmr_retriever | format_docs,
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =========================================================
# 4. Chat History Management & Interface
# =========================================================
# Initialize session state for storing past turn interactions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages on page rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process new user prompt
if user_query := st.chat_input("Ask a question regarding IS 800:2007..."):
    # Display user input in UI
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Format previous turn messages for LangChain chain history
    formatted_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_history.append(AIMessage(content=msg["content"]))

    # Generate assistant response with real-time streaming
    with st.chat_message("assistant"):
        response_container = st.empty()
        
        # Stream output chunk by chunk
        stream = rag_chain.stream({
            "question": user_query,
            "chat_history": formatted_history
        })
        
        full_response = response_container.write_stream(stream)

    # Save final assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
