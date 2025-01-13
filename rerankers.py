import os
import sys
from typing import List

import os
from dotenv import load_dotenv
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SentenceTransformerRerank


import pymupdf
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.core.agent import AgentRunner
from llama_index.agent.coa import CoAAgentWorker
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.llms.groq import Groq
import warnings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os
load_dotenv()

warnings.filterwarnings("ignore")
class Rerankers:
    def __init__(self):
        load_dotenv()
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
        
    def get_cohere_reranker(self, top_n=5):
        return CohereRerank(
            api_key=os.getenv("COHERE_API_KEY"),
            top_n=top_n
        )
    
    def get_colbert_reranker(self, top_n=5):
        return ColbertRerank(
            top_n=top_n,
            model="colbert-ir/colbertv2.0",
            tokenizer="colbert-ir/colbertv2.0",
            keep_retrieval_score=True
        )
    
    def get_sentence_transformer_reranker(self, top_n=5):
        return SentenceTransformerRerank(
            model="sentence-transformers/all-MiniLM-L6-v2",
            top_n=top_n
        )