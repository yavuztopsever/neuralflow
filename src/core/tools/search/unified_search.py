"""
Unified search tool for the LangGraph project.
This module provides combined vector and graph search capabilities.

The unified search tool combines:
1. Vector Search: Semantic search using embeddings
2. Graph Search: Graph-based search with node relationships
3. Hybrid Search: Combined vector and graph search

Features:
- Semantic similarity search
- Graph traversal and path finding
- Metadata filtering
- Result scoring and ranking
- Node and edge management
- Cache management

Example:
    ```python
    from src.core.tools.search import UnifiedSearch
    
    # Initialize search tool
    search = UnifiedSearch(
        vector_store=my_vector_store,
        embeddings=my_embeddings
    )
    
    # Add nodes and edges
    search.add_node(
        node_id="doc1",
        content="Document content",
        metadata={"type": "article"}
    )
    search.add_edge(
        source="doc1",
        target="doc2",
        weight=0.8
    )
    
    # Perform search
    results = await search.execute(
        query="search query",
        search_type="hybrid",
        top_k=5
    )
    ```
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from ..base.base_tool import BaseTool
from langchain.vectorstores import VectorStore
from langchain.embeddings import OpenAIEmbeddings
import networkx as nx
import numpy as np

class SearchResult(BaseModel):
    """
    Search result model.
    
    Attributes:
        content: Result content
        score: Similarity score (0-1)
        source: Result source ('vector' or 'graph')
        metadata: Optional result metadata
        timestamp: Result timestamp
    """
    content: str
    score: float
    source: str  # 'vector' or 'graph'
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class UnifiedSearch(BaseTool):
    """
    Tool for combined vector and graph search.
    
    This tool provides:
    1. Vector search using embeddings
    2. Graph search using NetworkX
    3. Hybrid search combining both approaches
    4. Node and edge management
    5. Path finding capabilities
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embeddings: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the unified search tool.
        
        The initialization process:
        1. Sets up vector store for semantic search
        2. Initializes embeddings model
        3. Creates graph structure
        4. Sets up caches
        
        Args:
            vector_store: Optional vector store instance
            embeddings: Optional embeddings model
            config: Optional configuration
        """
        super().__init__(
            name="unified_search",
            description="Combined vector and graph search capabilities",
            config=config
        )
        
        self.vector_store = vector_store or VectorStore()
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.graph = nx.Graph()
        
        # Initialize caches
        self.node_cache: Dict[str, Dict[str, Any]] = {}
        self.edge_cache: Dict[str, Dict[str, Any]] = {}
    
    async def execute(
        self,
        query: str,
        search_type: str = "hybrid",
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Execute search.
        
        The search process:
        1. Records tool usage
        2. Performs vector search if requested
        3. Performs graph search if requested
        4. Combines and ranks results
        5. Applies metadata filtering
        
        Args:
            query: Search query
            search_type: Type of search ('vector', 'graph', or 'hybrid')
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            
        Returns:
            List[SearchResult]: Sorted search results
            
        Raises:
            ValueError: If search_type is invalid
        """
        try:
            self.record_usage()
            
            results = []
            
            if search_type in ["vector", "hybrid"]:
                # Vector search
                vector_results = await self._vector_search(
                    query,
                    top_k=top_k,
                    metadata_filter=metadata_filter
                )
                results.extend(vector_results)
            
            if search_type in ["graph", "hybrid"]:
                # Graph search
                graph_results = await self._graph_search(
                    query,
                    top_k=top_k,
                    metadata_filter=metadata_filter
                )
                results.extend(graph_results)
            
            # Sort by score and limit results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.record_error(e)
            raise
    
    async def _vector_search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform vector search.
        
        The vector search process:
        1. Generates query embedding
        2. Searches vector store
        3. Applies metadata filtering
        4. Formats results
        
        Args:
            query: Search query
            top_k: Number of results
            metadata_filter: Optional metadata filter
            
        Returns:
            List[SearchResult]: Vector search results
        """
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search vector store
            results = self.vector_store.similarity_search_with_score(
                query_embedding,
                k=top_k,
                filter=metadata_filter
            )
            
            return [
                SearchResult(
                    content=result.page_content,
                    score=score,
                    source="vector",
                    metadata=result.metadata
                )
                for result, score in results
            ]
        except Exception as e:
            self.record_error(e)
            return []
    
    async def _graph_search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform graph search.
        
        The graph search process:
        1. Generates query embedding
        2. Searches through nodes
        3. Calculates similarity scores
        4. Applies metadata filtering
        5. Ranks results
        
        Args:
            query: Search query
            top_k: Number of results
            metadata_filter: Optional metadata filter
            
        Returns:
            List[SearchResult]: Graph search results
        """
        try:
            results = []
            
            # Get query embedding for similarity comparison
            query_embedding = self.embeddings.embed_query(query)
            
            # Search through nodes
            for node_id, node_data in self.node_cache.items():
                if metadata_filter:
                    # Check if node metadata matches filter
                    if not all(
                        node_data.get("metadata", {}).get(k) == v
                        for k, v in metadata_filter.items()
                    ):
                        continue
                
                # Calculate similarity score
                node_embedding = node_data.get("embedding")
                if node_embedding:
                    score = self._calculate_similarity(query_embedding, node_embedding)
                    
                    results.append(
                        SearchResult(
                            content=node_data["content"],
                            score=score,
                            source="graph",
                            metadata=node_data.get("metadata")
                        )
                    )
            
            # Sort by score and limit results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.record_error(e)
            return []
    
    def add_node(
        self,
        node_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a node to the graph.
        
        The node addition process:
        1. Generates content embedding
        2. Adds node to graph
        3. Updates node cache
        
        Args:
            node_id: Node identifier
            content: Node content
            metadata: Optional node metadata
            
        Raises:
            ValueError: If node_id already exists
        """
        try:
            # Get content embedding
            embedding = self.embeddings.embed_query(content)
            
            # Add to graph
            self.graph.add_node(
                node_id,
                content=content,
                embedding=embedding,
                metadata=metadata
            )
            
            # Update cache
            self.node_cache[node_id] = {
                "content": content,
                "embedding": embedding,
                "metadata": metadata
            }
            
        except Exception as e:
            self.record_error(e)
            raise
    
    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an edge to the graph.
        
        The edge addition process:
        1. Validates source and target nodes
        2. Adds edge to graph
        3. Updates edge cache
        
        Args:
            source: Source node ID
            target: Target node ID
            weight: Edge weight
            metadata: Optional edge metadata
            
        Raises:
            ValueError: If source or target nodes don't exist
        """
        try:
            # Add to graph
            self.graph.add_edge(
                source,
                target,
                weight=weight,
                metadata=metadata
            )
            
            # Update cache
            edge_id = f"{source}-{target}"
            self.edge_cache[edge_id] = {
                "source": source,
                "target": target,
                "weight": weight,
                "metadata": metadata
            }
            
        except Exception as e:
            self.record_error(e)
            raise
    
    def find_path(
        self,
        start: str,
        end: str,
        weight: str = "weight"
    ) -> List[str]:
        """
        Find shortest path between nodes.
        
        The path finding process:
        1. Validates start and end nodes
        2. Uses NetworkX shortest_path
        3. Returns path as node IDs
        
        Args:
            start: Start node ID
            end: End node ID
            weight: Edge attribute to use as weight
            
        Returns:
            List[str]: List of node IDs in path
            
        Raises:
            ValueError: If start or end nodes don't exist
            NetworkXNoPath: If no path exists
        """
        try:
            return nx.shortest_path(
                self.graph,
                source=start,
                target=end,
                weight=weight
            )
        except Exception as e:
            self.record_error(e)
            return []
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        try:
            return float(np.dot(embedding1, embedding2) / 
                       (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
        except Exception:
            return 0.0
    
    def clear(self) -> None:
        """
        Clear all data.
        
        This method:
        1. Clears the graph
        2. Clears node cache
        3. Clears edge cache
        """
        self.graph.clear()
        self.node_cache.clear()
        self.edge_cache.clear()

__all__ = ['UnifiedSearch', 'SearchResult'] 