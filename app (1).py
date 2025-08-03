import os
import cassio
import requests
import streamlit as st
from typing import List
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing_extensions import TypedDict
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# FastAPI Instance
app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic Model
class QueryInput(BaseModel):
    question: str

# Environment Variables
ASTRA_DB_TOKEN = os.environ.get('ASTRA_DB_TOKEN')
ASTRA_DB_ID = os.environ.get('ASTRA_DB_ID')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
HF_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

# Initialize AstraDB
cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)

# Document Loading
urls = [
    'https://www.opentext.com/what-is/agentic-ai',
    'https://dev.to/ajmal_hasan/genai-building-rag-systems-with-langchain-4dbp',
    'https://hatchworks.com/blog/ai-agents/ai-agents-explained/'
]
from langchain_community.document_loaders import WebBaseLoader
docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]

# Document Splitting
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='cl100k_base', chunk_size=500, chunk_overlap=0
)
docs_split = text_splitter.split_documents(doc_list)

# Embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_API_TOKEN
)

# Vector Store
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name='Vector_table',
    session=None,
    keyspace=None
)
astra_vector_store.add_documents(docs_split)
retriever = astra_vector_store.as_retriever()

# Wikipedia Tool
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_api)

# Router Model
class RouteQ(BaseModel):
    datasources: str  # 'vectorstore' or 'wiki_search'

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name='llama-3.3-70b-versatile')
structured_llm_router = llm.with_structured_output(RouteQ)

system = "You are an expert router. If the question is about agents, prompt engineering or adversarial attacks, use vectorstore, else wiki-search."
route_prompt = ChatPromptTemplate.from_messages([
    ('system', system),
    ('human', '{question}')
])
question_router = route_prompt | structured_llm_router

# LangGraph State
class GraphState(TypedDict):
    question: str
    documents: List[str]

def retrieve(state):
    question = state['question']
    documents = retriever.invoke(question)
    return {'documents': documents, 'question': question}

def wiki_search(state):
    question = state['question']
    docs = wiki.invoke({"query": question})
    wiki_result = Document(page_content=docs)
    return {'documents': wiki_result, 'question': question}

import logging
logging.basicConfig(level=logging.INFO)

def route_question(state):
    question = state['question']
    source = question_router.invoke({"question": question})
    logging.info(f"Routing Decision: {source.datasources}")
    if source.datasources == "wiki_search":
        return "wiki_search"
    elif source.datasources == "vectorstore":
        return "vectorstore"
    else:
        logging.warning(f"Routing failed, defaulting to retrieve.")
        return "wiki_search"


workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)
workflow.add_conditional_edges(START, route_question, {"wiki_search": "wiki_search", "vectorstore": "retrieve"})
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)
graph_app = workflow.compile()

# FastAPI Root
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangGraph Multi-Agent Bot</title>
    </head>
    <body>
        <h1>LangGraph Multi-Agent Bot</h1>
        <form action="/ask" method="post">
            <input type="text" name="question" placeholder="Enter your question" required>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """

# API Endpoint
from fastapi import Request

from fastapi import Form

from fastapi.responses import HTMLResponse, JSONResponse

@app.post("/ask")
async def ask_question(request: Request):
    content_type = request.headers.get('content-type')
    question = None

    if 'application/json' in content_type:
        data = await request.json()
        question = data.get("question")
    elif 'application/x-www-form-urlencoded' in content_type:
        form_data = await request.form()
        question = form_data.get("question")

    if not question:
        return HTMLResponse("<h2>No question provided.</h2>")

    inputs = {"question": question}
    final_result = None
    for output in graph_app.stream(inputs):
        for key, value in output.items():
            final_result = value

    docs = final_result['documents']
    result_text = [doc.page_content for doc in docs] if isinstance(docs, list) else [docs.page_content]

    # Smart Response: If frontend is HTML form, return HTML; else return JSON
    if 'application/json' in content_type:
        return JSONResponse(content={"response": result_text[0]})
    else:
        return HTMLResponse(f"<h2>Question: {question}</h2><p>{result_text[0]}</p><a href='/'>Ask another</a>")



