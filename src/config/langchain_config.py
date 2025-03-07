"""
Configuration settings for LangChain integration.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub, Cohere
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, Pinecone, Weaviate
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory
)
from langchain.chains import (
    LLMChain,
    SequentialChain,
    SimpleSequentialChain,
    RetrievalQA,
    ConversationalRetrievalChain
)
from langchain.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
    ChatPromptTemplate
)
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools import Tool
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: str = "openai"  # openai, huggingface, cohere
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: str
    api_base: Optional[str] = None
    api_version: Optional[str] = None

class MemoryConfig(BaseModel):
    """Configuration for memory management."""
    type: str = "buffer"  # buffer, summary, window, token
    max_items: int = 10
    max_tokens: int = 2000
    return_messages: bool = True

class VectorStoreConfig(BaseModel):
    """Configuration for vector stores."""
    type: str = "chroma"  # chroma, faiss, pinecone, weaviate
    persist_directory: str = "./data/vectorstore"
    api_key: Optional[str] = None
    environment: Optional[str] = None
    index_name: Optional[str] = None

class DocumentLoaderConfig(BaseModel):
    """Configuration for document loaders."""
    type: str = "text"  # text, pdf, csv, web
    file_path: Optional[str] = None
    url: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200

class LangChainConfig(BaseModel):
    """Configuration for LangChain components."""
    llm: LLMConfig
    memory: MemoryConfig
    vector_store: VectorStoreConfig
    document_loader: Optional[DocumentLoaderConfig] = None

class LangChainManager:
    """Manager class for LangChain components."""
    
    def __init__(self, config: LangChainConfig):
        self.config = config
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all LangChain components."""
        # Initialize LLM
        self.llm = self._create_llm()
        
        # Initialize embeddings
        self.embeddings = self._create_embeddings()
        
        # Initialize vector store
        self.vector_store = self._create_vector_store()
        
        # Initialize memory
        self.memory = self._create_memory()
        
        # Initialize document loader if configured
        if self.config.document_loader:
            self.document_loader = self._create_document_loader()
    
    def _create_llm(self):
        """Create LLM based on configuration."""
        if self.config.llm.provider == "openai":
            return ChatOpenAI(
                model_name=self.config.llm.model_name,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                openai_api_key=self.config.llm.api_key,
                openai_api_base=self.config.llm.api_base,
                openai_api_version=self.config.llm.api_version
            )
        elif self.config.llm.provider == "huggingface":
            return HuggingFaceHub(
                repo_id=self.config.llm.model_name,
                model_kwargs={"temperature": self.config.llm.temperature},
                huggingfacehub_api_token=self.config.llm.api_key
            )
        elif self.config.llm.provider == "cohere":
            return Cohere(
                model=self.config.llm.model_name,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                cohere_api_key=self.config.llm.api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")
    
    def _create_embeddings(self):
        """Create embeddings based on configuration."""
        if self.config.llm.provider == "openai":
            return OpenAIEmbeddings(
                openai_api_key=self.config.llm.api_key
            )
        else:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
    
    def _create_vector_store(self):
        """Create vector store based on configuration."""
        if self.config.vector_store.type == "chroma":
            return Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.config.vector_store.persist_directory
            )
        elif self.config.vector_store.type == "faiss":
            return FAISS.from_texts(
                texts=[],
                embedding=self.embeddings
            )
        elif self.config.vector_store.type == "pinecone":
            import pinecone
            pinecone.init(
                api_key=self.config.vector_store.api_key,
                environment=self.config.vector_store.environment
            )
            return Pinecone.from_texts(
                texts=[],
                embedding=self.embeddings,
                index_name=self.config.vector_store.index_name
            )
        elif self.config.vector_store.type == "weaviate":
            return Weaviate.from_texts(
                texts=[],
                embedding=self.embeddings,
                weaviate_client_config={
                    "url": self.config.vector_store.persist_directory,
                    "api_key": self.config.vector_store.api_key
                }
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.config.vector_store.type}")
    
    def _create_memory(self):
        """Create memory based on configuration."""
        if self.config.memory.type == "buffer":
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=self.config.memory.return_messages
            )
        elif self.config.memory.type == "summary":
            return ConversationSummaryMemory(
                llm=self.llm,
                memory_key="summary"
            )
        elif self.config.memory.type == "window":
            return ConversationBufferWindowMemory(
                k=self.config.memory.max_items,
                memory_key="chat_history",
                return_messages=self.config.memory.return_messages
            )
        elif self.config.memory.type == "token":
            return ConversationTokenBufferMemory(
                llm=self.llm,
                max_token_limit=self.config.memory.max_tokens,
                memory_key="chat_history",
                return_messages=self.config.memory.return_messages
            )
        else:
            raise ValueError(f"Unsupported memory type: {self.config.memory.type}")
    
    def _create_document_loader(self):
        """Create document loader based on configuration."""
        if self.config.document_loader.type == "text":
            return TextLoader(self.config.document_loader.file_path)
        elif self.config.document_loader.type == "pdf":
            return PyPDFLoader(self.config.document_loader.file_path)
        elif self.config.document_loader.type == "csv":
            return CSVLoader(self.config.document_loader.file_path)
        elif self.config.document_loader.type == "web":
            return WebBaseLoader(self.config.document_loader.url)
        else:
            raise ValueError(f"Unsupported document loader type: {self.config.document_loader.type}")
    
    def create_chain(self, prompt_template: str) -> LLMChain:
        """Create a new LLMChain with the given prompt template."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("human", "{input}"),
            ("ai", "Let me help you with that.")
        ])
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory
        )
    
    def create_sequential_chain(self, chains: list[LLMChain]) -> SequentialChain:
        """Create a SequentialChain from a list of LLMChains."""
        return SequentialChain(
            chains=chains,
            input_variables=["input"],
            output_variables=["response"]
        )
    
    def create_simple_sequential_chain(self, chains: list[LLMChain]) -> SimpleSequentialChain:
        """Create a SimpleSequentialChain from a list of LLMChains."""
        return SimpleSequentialChain(chains=chains)
    
    def create_retrieval_qa_chain(self) -> RetrievalQA:
        """Create a RetrievalQA chain."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
    
    def create_conversational_retrieval_chain(self) -> ConversationalRetrievalChain:
        """Create a ConversationalRetrievalChain."""
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory
        )
    
    def create_agent(self, tools: list[Tool], agent_type: str = "zero_shot") -> AgentExecutor:
        """Create an agent with the given tools."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following tools to help answer the user's question:\n\n{tools}\n\nUse the following format:\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question"),
            ("human", "{input}"),
            ("assistant", "Let me approach this step by step:"),
            ("human", "{agent_scratchpad}")
        ])
        
        llm_with_stop = self.llm.bind(stop=["\nObservation:"])
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
                "tools": lambda x: "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                "tool_names": lambda x: ", ".join([tool.name for tool in tools])
            }
            | prompt
            | llm_with_stop
            | ReActSingleInputOutputParser()
        )
        
        return AgentExecutor(agent=agent, tools=tools, verbose=True) 