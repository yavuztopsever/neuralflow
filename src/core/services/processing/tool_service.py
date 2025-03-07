"""
Tool service for the LangGraph project.
This service provides tool management capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

from ..tools.langchain_tools import LangChainTools
from langchain.tools import Tool

class ToolService:
    """Service for managing tools in the LangGraph system."""
    
    def __init__(self):
        """Initialize the tool service."""
        self.tools: Dict[str, LangChainTools] = {}
        self.history: List[Dict[str, Any]] = []
    
    def create_tools(self, name: str, vector_store: Any, llm: Any) -> LangChainTools:
        """
        Create a new tools instance.
        
        Args:
            name: Tools name
            vector_store: Vector store for search
            llm: Language model
            
        Returns:
            LangChainTools: Created tools instance
        """
        tools = LangChainTools(vector_store=vector_store, llm=llm)
        self.tools[name] = tools
        return tools
    
    def get_tools(self, name: str) -> Optional[LangChainTools]:
        """
        Get a tools instance by name.
        
        Args:
            name: Tools name
            
        Returns:
            Optional[LangChainTools]: Tools instance if found, None otherwise
        """
        return self.tools.get(name)
    
    def get_all_tools(
        self,
        name: str,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        db_url: Optional[str] = None
    ) -> List[Tool]:
        """
        Get all available tools.
        
        Args:
            name: Tools name
            google_api_key: Optional Google API key
            google_cse_id: Optional Google CSE ID
            db_url: Optional database URL
            
        Returns:
            List[Tool]: List of tools
        """
        tools_instance = self.get_tools(name)
        if not tools_instance:
            raise ValueError(f"Tools not found: {name}")
        
        try:
            tools = tools_instance.get_all_tools(
                google_api_key=google_api_key,
                google_cse_id=google_cse_id,
                db_url=db_url
            )
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now(),
                'tools': name,
                'action': 'get_all_tools',
                'tool_count': len(tools)
            })
            
            return tools
            
        except Exception as e:
            # Record error in history
            self.history.append({
                'timestamp': datetime.now(),
                'tools': name,
                'action': 'get_all_tools',
                'error': str(e)
            })
            raise
    
    def get_search_tool(self, name: str) -> Tool:
        """
        Get the search tool.
        
        Args:
            name: Tools name
            
        Returns:
            Tool: Search tool
        """
        tools_instance = self.get_tools(name)
        if not tools_instance:
            raise ValueError(f"Tools not found: {name}")
        
        try:
            tool = tools_instance.get_search_tool()
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now(),
                'tools': name,
                'action': 'get_search_tool'
            })
            
            return tool
            
        except Exception as e:
            # Record error in history
            self.history.append({
                'timestamp': datetime.now(),
                'tools': name,
                'action': 'get_search_tool',
                'error': str(e)
            })
            raise
    
    def get_qa_tool(self, name: str) -> Tool:
        """
        Get the QA tool.
        
        Args:
            name: Tools name
            
        Returns:
            Tool: QA tool
        """
        tools_instance = self.get_tools(name)
        if not tools_instance:
            raise ValueError(f"Tools not found: {name}")
        
        try:
            tool = tools_instance.get_qa_tool()
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now(),
                'tools': name,
                'action': 'get_qa_tool'
            })
            
            return tool
            
        except Exception as e:
            # Record error in history
            self.history.append({
                'timestamp': datetime.now(),
                'tools': name,
                'action': 'get_qa_tool',
                'error': str(e)
            })
            raise
    
    def get_web_crawl_tool(self, name: str) -> Tool:
        """
        Get the web crawler tool.
        
        Args:
            name: Tools name
            
        Returns:
            Tool: Web crawler tool
        """
        tools_instance = self.get_tools(name)
        if not tools_instance:
            raise ValueError(f"Tools not found: {name}")
        
        try:
            tool = tools_instance.get_web_crawl_tool()
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now(),
                'tools': name,
                'action': 'get_web_crawl_tool'
            })
            
            return tool
            
        except Exception as e:
            # Record error in history
            self.history.append({
                'timestamp': datetime.now(),
                'tools': name,
                'action': 'get_web_crawl_tool',
                'error': str(e)
            })
            raise
    
    def get_history(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the tool usage history.
        
        Args:
            name: Optional tools name to filter history
            
        Returns:
            List[Dict[str, Any]]: Tool usage history
        """
        if name:
            return [entry for entry in self.history if entry.get('tools') == name]
        return self.history
    
    def clear_history(self, name: Optional[str] = None) -> None:
        """
        Clear the tool usage history.
        
        Args:
            name: Optional tools name to clear history for
        """
        if name:
            self.history = [entry for entry in self.history if entry.get('tools') != name]
        else:
            self.history = []
    
    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset tools.
        
        Args:
            name: Optional tools name to reset
        """
        if name:
            self.tools.pop(name, None)
        else:
            self.tools = {}
            self.history = []

__all__ = ['ToolService'] 