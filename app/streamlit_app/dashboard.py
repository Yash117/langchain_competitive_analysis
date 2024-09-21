import streamlit as st
from langchain.chat_models import ChatOpenAI

from app.langgraph_agent.langgraph_chatbot import LangGraphChatbot
from app.models.colbert_model import ColbertModel
from app.models.dpr_model import DprModel
from app.models.splade_model import SpladeModel
from app.models.bm25_bert_model import BM25BertModel
from app.models.ance_model import AnceModel
from app.models.minilm_model import MinilmModel
from app.models.reformer_model import ReformerModel
from app.rag_pipeline.iterative_rag import IterativeRag
from app.rag_pipeline.retriever import Retriever

# Initialize session state for storing user inputs
if 'industry' not in st.session_state:
    st.session_state.industry = ''
if 'competitors' not in st.session_state:
    st.session_state.competitors = ''
if 'focus_areas' not in st.session_state:
    st.session_state.focus_areas = []
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'insights' not in st.session_state:
    st.session_state.insights = ''

st.title('AI-Powered Competitive Analysis for Startups')

# Sidebar for user inputs and model selection
with st.sidebar:
    st.header('Configure Analysis')
    st.session_state.industry = st.text_input('Your Industry', placeholder='e.g., Technology, Healthcare')
    st.session_state.competitors = st.text_input('Competitor Names', placeholder='e.g., Google, Microsoft')
    st.session_state.focus_areas = st.multiselect(
        'Key Focus Areas',
        ['Pricing', 'Market Share', 'Growth Strategy', 'Funding'],
    )
    retrieval_model = st.selectbox(
        'Choose Retrieval Model', 
        ['ColBERT', 'SPLADE', 'DPR', 'BM25-BERT Hybrid', 'ANCE', 'MiniLM', 'Reformer']
    )

# Main area for chat interaction and insights display
st.header('Competitive Insights')

# Initialize retrieval models based on user selection
colbert_model = ColbertModel(
    colbert_model_name="klue/colbert-base",
    index_path="./colbert_index",  # Replace with the actual path to your ColBERT index
)
splade_model = SpladeModel(
    bert_model_name='bert-base-uncased',  # Replace with a suitable BERT model
)
dpr_model = DprModel(
    dpr_model_name="facebook/dpr-ctx_encoder-single-nq-base",
    index_path="./dpr_index",  # Replace with the actual path to your DPR index
)
bm25_bert_model = BM25BertModel(bert_model_name="bert-base-uncased")
ance_model = AnceModel(
    ance_model_name="sentence-transformers/all-mpnet-base-v2",
    index_path="./ance_index",
)
minilm_model = MinilmModel(
    minilm_model_name="sentence-transformers/all-MiniLM-L6-v2",
    index_path="./minilm_index",
)
reformer_model = ReformerModel(
    reformer_model_name="google/reformer-enwik8",
    index_path="./reformer_index",
)

retriever = Retriever(models={
    "ColBERT": colbert_model,
    "SPLADE": splade_model,
    "DPR": dpr_model,
    "BM25-BERT Hybrid": bm25_bert_model,
    "ANCE": ance_model,
    "MiniLM": minilm_model,
    "Reformer": reformer_model,
})

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Initialize LangGraph agent and RAG pipeline
langgraph_agent = LangGraphChatbot(llm=llm)
rag_pipeline = IterativeRag(retriever=retriever, llm=llm)

# Chat input area
user_input = st.text_input('Ask a question:', '')

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Process user input and refine query with LangGraph agent
    refined_query = langgraph_agent.get_refined_query(
        st.session_state.messages,
        industry=st.session_state.industry,
        competitors=st.session_state.competitors,
        focus_areas=st.session_state.focus_areas,
    )
    # Get competitive insights using RAG
    st.session_state.insights = rag_pipeline.get_competitive_insights(
        query=refined_query,
        focus_areas=st.session_state.focus_areas,
        model_name=retrieval_model
    )
    st.session_state.messages.append({"role": "assistant", "content": st.session_state.insights['insights']})

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
