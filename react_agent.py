import os
import sys
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from COA_agent import FusionRetrieval

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.core.agent import AgentRunner
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.llms.groq import Groq
# from ReRanker.rerankers import Rerankers
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryPlanTool
from llama_index.core import get_response_synthesizer
from llama_index.core.memory import (
    VectorMemory,
    SimpleComposableMemory,
    ChatMemoryBuffer,
)


class React_Agent:
    def __init__(self):
        self.query_engine_tools = []
        self._setup_models()
        #self.reranker = Rerankers()
        #self.colbert_reranker = self.reranker.get_colbert_reranker()
        self.response_synthesizer = get_response_synthesizer()
        self.query_plan_tool = None
        

    def _setup_models(self) -> None:
        """Initialize LLM and embedding models"""
        Settings.llm = Groq(
            model="llama3-8b-8192", 
            temperature=0.0, 
            api_key=os.getenv("GROQ_API_KEY")
        )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.llm = Settings.llm


    def setup_tool(self, base_folder_path: str) -> None:
        
                documents = SimpleDirectoryReader(base_folder_path).load_data()
                
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

    def setup_query_plan_tool(self) -> None:
        self.query_plan_tool = QueryPlanTool.from_defaults(
            query_engine_tools=self.query_engine_tools,
            response_synthesizer=self.response_synthesizer,
        )
    def setup_web_search_tool(self) -> None:
        tavily_tool = TavilyToolSpec(
            api_key=os.getenv("TAVILY_API_KEY"),
            
        )
        self.query_engine_tools = tavily_tool.to_tool_list
        
    def print_tools(self) -> None:
        for tool in self.query_engine_tools:
            if hasattr(tool, 'metadata') and tool.metadata.name is not None:
                print(f"Tool Name: {tool.metadata.name}")
                print(f"Description: {tool.metadata.description}\n")

    def setup_memory(self) -> None:
        vector_memory = VectorMemory.from_defaults(
            vector_store=None,
            embed_model=Settings.embed_model,
            retriever_kwargs={"similarity_top_k": 5},
        )
        chat_memory_buffer = ChatMemoryBuffer.from_defaults()

        self.memory = SimpleComposableMemory.from_defaults(
            primary_memory=chat_memory_buffer,
            secondary_memory_sources=[vector_memory],
        )    
          

    def setup_agent(self):
       agent = ReActAgent.from_tools(
            self.query_engine_tools,
            llm=self.llm,
            max_function_calls=10,
            memory = self.memory,
            verbose=True,
        )
       return agent

    # def memory(self):
    #     vector_memory = VectorMemory.from_defaults(
    #     vector_store=None,  # leave as None to use default in-memory vector store
    #     embed_model=Oembed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    #     retriever_kwargs={"similarity_top_k": 2},
    #     )

    #     chat_memory_buffer = ChatMemoryBuffer.from_defaults()

    #     composable_memory = SimpleComposableMemory.from_defaults(
    #     primary_memory=chat_memory_buffer,
    #     secondary_memory_sources=[vector_memory],
    #     )  
    #     llm = OpenAI(model="gpt-3.5-turbo-0613")

    #     composable_memory = SimpleComposableMemory.from_defaults(
    #     primary_memory=ChatMemoryBuffer.from_defaults(),
    #     secondary_memory_sources=[
    #     vector_memory.copy(
    #         deep=True
    #     )  # using a copy here for illustration purposes
    #     # later will use original vector_memory again
    #     ],
    #     )
    #     agent = ReActAgent.from_tools(
    #         self.query_engine_tools,
    #         llm=self.llm,
    #         max_function_calls=10,
    #         memory = self.memory,
    #         verbose=True,
    #     )
    #    return agent
   

   

def main():
    agent_instance = React_Agent()

    base_data_path = "Data"  
    agent_instance.setup_tool(base_folder_path=base_data_path)

    agent_instance.setup_memory()

    function_calling_agent = agent_instance.setup_agent()

    question = "How is the performance (like accuracy or similar metrics) of the methodlogy of Segmented Video Activity Recognition in the paper titled 'Detailed Action Identification in Baseball Game Recordings'?"
    response = function_calling_agent.chat(question)
    
    print(str(response))

    print(function_calling_agent.chat_history)

if __name__ == "__main__":
    main()