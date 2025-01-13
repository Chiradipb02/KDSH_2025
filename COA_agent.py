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

class FusionRetrieval:
    def __init__(self, splitter=None, chunk_size=256, vector_top_k=5, bm25_top_k=10, fusion_weights=[0.6, 0.4], fusion_top_k=10, mode="relative_score", num_queries=1, use_async=True, verbose=False):
        embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embedding_model
        Settings.llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
        self.splitter = splitter or SentenceSplitter(chunk_size=chunk_size)
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.fusion_weights = fusion_weights
        self.fusion_top_k = fusion_top_k
        self.mode = mode
        self.num_queries = num_queries
        self.use_async = use_async
        self.verbose = verbose
        self.index = None
        self.retriever = None

    def build_index(self, documents):
        self.index = VectorStoreIndex.from_documents(
            documents, transformations=[self.splitter], show_progress=True
        )

    def setup_relative_score_retriever(self):
        if not self.index:
            raise ValueError("Index has not been built. Call build_index() first.")
        
        vector_retriever = self.index.as_retriever(similarity_top_k=self.vector_top_k)
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.index.docstore, similarity_top_k=self.bm25_top_k
        )
        self.retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            retriever_weights=self.fusion_weights,
            similarity_top_k=self.fusion_top_k,
            num_queries=self.num_queries,
            mode=self.mode,
            use_async=self.use_async,
            verbose=self.verbose,
        )
    def setup_distribution_based_retriever(self):
        vector_retriever = self.index.as_retriever(similarity_top_k=self.vector_top_k)
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.index.docstore, similarity_top_k=self.bm25_top_k
        )
        self.retriever =  QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            retriever_weights=self.fusion_weights,
            similarity_top_k=self.fusion_top_k,
            num_queries=self.num_queries,
            mode="dist_based_score",
            use_async=self.use_async,
            verbose=self.verbose,
        )
    
    def setup_reciprocal_reranking_retriever(self):
        vector_retriever = self.index.as_retriever(similarity_top_k=self.vector_top_k)
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.index.docstore, similarity_top_k=self.bm25_top_k
        )
        self.retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=self.fusion_top_k,
            num_queries=self.num_queries,
            mode="reciprocal_rerank",
            use_async=self.use_async,
            verbose=self.verbose,
        )
        

    def retrieve(self, query):

        if not self.retriever:
            raise ValueError("Retrievers have not been set up. Call setup_retrievers() first.")
        
        return self.retriever.retrieve(query)




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
class RetrievalAgent:
    def __init__(self):
        self.query_engine_tools = []
        self._setup_models()
        self.reranker = Rerankers()
        self.colbert_reranker = self.reranker.get_colbert_reranker()
        

    def _setup_models(self) -> None:
        """Initialize LLM and embedding models"""
        Settings.llm = Groq(
            model="llama3-70b-8192", 
            temperature=0.0, 
            api_key=os.getenv("GROQ_API_KEY")
        )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

    def setup_tool(self, folder_path: str) -> None:
        
        
        
            documents = SimpleDirectoryReader(folder_path).load_data()
                
            fusion = FusionRetrieval(
                chunk_size=256, 
                fusion_weights=[0.6, 0.4], 
                verbose=True
            )
            fusion.build_index(documents)
            fusion.setup_relative_score_retriever()
                
            query_engine = RetrieverQueryEngine.from_args(
                retriever=fusion.retriever
            )
                
            self.query_engine_tools.append(
            QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name=f"vector_tool",
                    description="A query engine tool for the extracting specific informations like tables, performance metrics of the methodlogy in the paper.",
                ),
            )
        )


    def print_tools(self) -> None:
        for tool in self.query_engine_tools:
            if hasattr(tool, 'metadata') and tool.metadata.name is not None:
                print(f"Tool Name: {tool.metadata.name}")
                print(f"Description: {tool.metadata.description}\n")

    def setup_agent(self) -> CoAAgentWorker:
        return CoAAgentWorker.from_tools(
            tools=self.query_engine_tools,
            llm=Settings.llm,
            verbose=True
        )

def main():
    agent_retrieval = RetrievalAgent()
    
    agent_retrieval.setup_tool('Data')
    
    agent_retrieval.print_tools()
    
    worker = agent_retrieval.setup_agent()
    agent = worker.as_agent()
    
    answer = agent.chat("How is the performance (like accuracy or similar metrics) of the methodlogy in the paper titled 'Detailed Action Identification in Baseball Game Recordings'?")
    print("\nAgent Response:")
    print(str(answer))

if __name__ == "__main__":
    main()