"""
LangChain tools implementation.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import Tool, StructuredTool
from langchain.vectorstores import VectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.prompt import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain.utilities import (
    PythonREPLTool,
    RequestsWrapper,
    GoogleSearchAPIWrapper,
    WikipediaAPIWrapper,
    SQLDatabaseChain
)
from langchain.tools.python.tool import PythonREPLTool
from langchain.tools.requests.requests import RequestsWrapper
from langchain.tools.calculator.tool import CalculatorTool
from langchain.tools.google_search import GoogleSearchAPIWrapper
from langchain.tools.wikipedia import WikipediaAPIWrapper
from langchain.tools.sql_database.tool import SQLDatabaseTool
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.tools.sql_database.tool import InfoSQLDatabaseTool
from langchain.tools.sql_database.tool import ListSQLDatabaseTool
from langchain.tools.sql_database.tool import QuerySQLCheckerTool
import crawl4ai
from crawl4ai import WebCrawler, CrawlConfig

class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(..., description="The search query")
    k: int = Field(default=3, description="Number of results to return")

class QAToolInput(BaseModel):
    """Input schema for QA tool."""
    question: str = Field(..., description="The question to answer")
    context: Optional[str] = Field(None, description="Additional context for the question")

class WebCrawlInput(BaseModel):
    """Input schema for web crawling tool."""
    url: str = Field(..., description="The URL to crawl")
    max_pages: int = Field(default=5, description="Maximum number of pages to crawl")
    max_depth: int = Field(default=2, description="Maximum crawl depth")
    follow_links: bool = Field(default=True, description="Whether to follow links")
    extract_text: bool = Field(default=True, description="Whether to extract text content")

class LangChainTools:
    """Collection of LangChain tools."""
    
    def __init__(self, vector_store: VectorStore, llm: Any):
        self.vector_store = vector_store
        self.llm = llm
        
        # Initialize web crawler
        self.web_crawler = WebCrawler(
            config=CrawlConfig(
                max_pages=5,
                max_depth=2,
                follow_links=True,
                extract_text=True
            )
        )
        
        # Create structured prompts
        self.qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant that answers questions based on the provided context."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="Context: {context}\nQuestion: {question}")
        ])
        
        self.qa_chain = self.qa_prompt | self.llm | StrOutputParser()
    
    def get_search_tool(self) -> Tool:
        """Create a search tool using vector store with structured input."""
        return StructuredTool.from_function(
            func=self.vector_store.similarity_search,
            name="Search",
            description="Search for relevant information in the knowledge base",
            args_schema=SearchInput
        )
    
    def get_qa_tool(self) -> Tool:
        """Create a question-answering tool with structured input."""
        return StructuredTool.from_function(
            func=self._qa_tool_func,
            name="QA",
            description="Answer questions based on the knowledge base",
            args_schema=QAToolInput
        )
    
    def get_web_crawl_tool(self) -> Tool:
        """Create a web crawling tool using crawl4ai."""
        return StructuredTool.from_function(
            func=self._web_crawl_func,
            name="Web Crawler",
            description="Crawl websites and extract information",
            args_schema=WebCrawlInput
        )
    
    def _web_crawl_func(self, url: str, max_pages: int = 5, max_depth: int = 2, 
                       follow_links: bool = True, extract_text: bool = True) -> str:
        """Internal function for web crawling tool."""
        try:
            # Update crawler config
            self.web_crawler.config.max_pages = max_pages
            self.web_crawler.config.max_depth = max_depth
            self.web_crawler.config.follow_links = follow_links
            self.web_crawler.config.extract_text = extract_text
            
            # Perform crawl
            results = self.web_crawler.crawl(url)
            
            # Process results
            if extract_text:
                # Extract text content from crawled pages
                text_content = []
                for page in results.pages:
                    if page.text:
                        text_content.append(page.text)
                
                return "\n\n".join(text_content)
            else:
                # Return structured crawl results
                return str(results)
                
        except Exception as e:
            return f"Error during web crawling: {str(e)}"
    
    def _qa_tool_func(self, question: str, context: Optional[str] = None) -> str:
        """Internal function for QA tool."""
        # If no context provided, retrieve from vector store
        if not context:
            search_results = self.vector_store.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in search_results])
        
        # Use the QA chain
        return self.qa_chain.invoke({
            "question": question,
            "context": context,
            "chat_history": []
        })
    
    def get_python_tool(self) -> Tool:
        """Create a Python REPL tool with structured input."""
        return StructuredTool.from_function(
            func=PythonREPLTool().run,
            name="Python REPL",
            description="Execute Python code in a REPL environment",
            args_schema=BaseModel
        )
    
    def get_requests_tool(self) -> Tool:
        """Create a requests tool with structured input."""
        return StructuredTool.from_function(
            func=RequestsWrapper().run,
            name="HTTP Requests",
            description="Make HTTP requests to external APIs",
            args_schema=BaseModel
        )
    
    def get_calculator_tool(self) -> Tool:
        """Create a calculator tool with structured input."""
        return StructuredTool.from_function(
            func=CalculatorTool().run,
            name="Calculator",
            description="Perform mathematical calculations",
            args_schema=BaseModel
        )
    
    def get_google_search_tool(self, api_key: str, cse_id: str) -> Tool:
        """Create a Google search tool with structured input."""
        search = GoogleSearchAPIWrapper(google_api_key=api_key, google_cse_id=cse_id)
        return StructuredTool.from_function(
            func=search.run,
            name="Google Search",
            description="Search the internet using Google",
            args_schema=SearchInput
        )
    
    def get_wikipedia_tool(self) -> Tool:
        """Create a Wikipedia search tool with structured input."""
        wikipedia = WikipediaAPIWrapper()
        return StructuredTool.from_function(
            func=wikipedia.run,
            name="Wikipedia",
            description="Search Wikipedia for information",
            args_schema=SearchInput
        )
    
    def get_sql_tools(self, db_url: str) -> List[Tool]:
        """Create SQL database tools with structured input."""
        db = SQLDatabaseTool(db_url=db_url)
        return [
            StructuredTool.from_function(
                func=tool.run,
                name=tool.name,
                description=tool.description,
                args_schema=BaseModel
            )
            for tool in [
                QuerySQLDataBaseTool(db=db),
                InfoSQLDatabaseTool(db=db),
                ListSQLDatabaseTool(db=db),
                QuerySQLCheckerTool(db=db)
            ]
        ]
    
    def get_all_tools(self, google_api_key: str = None, google_cse_id: str = None, db_url: str = None) -> List[Tool]:
        """Get all available tools."""
        tools = [
            self.get_search_tool(),
            self.get_qa_tool(),
            self.get_web_crawl_tool(),  # Add web crawler tool
            self.get_python_tool(),
            self.get_requests_tool(),
            self.get_calculator_tool()
        ]
        
        if google_api_key and google_cse_id:
            tools.append(self.get_google_search_tool(google_api_key, google_cse_id))
        
        tools.append(self.get_wikipedia_tool())
        
        if db_url:
            tools.extend(self.get_sql_tools(db_url))
        
        return tools 