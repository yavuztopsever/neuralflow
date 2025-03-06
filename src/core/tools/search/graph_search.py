import os
import json
import networkx as nx
from cachetools import cached, TTLCache
from tools.memory_manager import MemoryManager
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from langchain.schema import Document
from langchain.vectorstores import VectorStore
from langchain.embeddings import OpenAIEmbeddings

@dataclass
class GraphNode:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float
    metadata: Optional[Dict[str, Any]] = None

class GraphSearch:
    """Handles knowledge graph-based search and traversal."""

    def __init__(self, config=None, memory_manager=None, graph_path=None, vector_store: Optional[VectorStore] = None):
        """
        Initializes the GraphSearch object.

        Args:
            config (Config, optional): Configuration object. If None, uses the global Config.
            memory_manager (MemoryManager, optional): An instance of MemoryManager for caching.
            graph_path (str, optional): Path to the knowledge graph file. Overrides config.GRAPH_PATH if provided.
            vector_store (VectorStore, optional): An instance of VectorStore for semantic search.
        """
        # Import Config here to avoid circular imports at module level
        from config.config import Config as GlobalConfig
        
        self.config = config or GlobalConfig
        self.graph_path = graph_path or getattr(self.config, 'GRAPH_PATH', 'graph_store/knowledge_graph.json')
        
        # Convert Path to string if needed
        if hasattr(self.graph_path, 'as_posix'):
            self.graph_path = self.graph_path.as_posix()
            
        self.memory_manager = memory_manager or MemoryManager()
        self.vector_store = vector_store or VectorStore()
        self.embeddings = OpenAIEmbeddings()
        self.node_cache: Dict[str, GraphNode] = {}
        self.edge_cache: Dict[Tuple[str, str], GraphEdge] = {}
        self.initialize()

    def initialize(self):
        """Initializes all necessary components."""
        self.graph = self._load_graph()
        self.relationship_cache = TTLCache(maxsize=1000, ttl=self.config.RELATIONSHIP_CACHE_TTL)  # Use config for TTL
        self.graph.state = {}  # Initialize the state attribute

    @cached(cache=lambda self: self.relationship_cache)
    def _get_relationship(self, node1, node2):
        """Gets the relationship between two nodes."""
        return self.graph.get_edge_data(node1, node2)

    def _load_graph(self):
        """
        Loads the knowledge graph from a JSON file or initializes a new one.

        Returns:
            networkx.DiGraph: The loaded or initialized knowledge graph.
        """
        # Always return a fresh DiGraph to avoid initialization errors
        # We'll try to load data into it if the file exists
        graph = nx.DiGraph()
        
        # Only try to load from file if it exists
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Extract nodes (should be present in any JSON graph format)
                    if 'nodes' in data:
                        for node_data in data['nodes']:
                            node_id = node_data.get('id', None)
                            if node_id is not None:
                                graph.add_node(node_id, **{k: v for k, v in node_data.items() if k != 'id'})
                    
                    # Extract edges/links (handle different naming conventions)
                    links_key = 'links' if 'links' in data else 'edges'
                    if links_key in data:
                        for edge_data in data[links_key]:
                            source = edge_data.get('source', None)
                            target = edge_data.get('target', None)
                            if source is not None and target is not None:
                                # Add all other attributes as edge attributes
                                attrs = {k: v for k, v in edge_data.items() 
                                        if k not in ('source', 'target')}
                                graph.add_edge(source, target, **attrs)
                
                print(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
                return graph
            except (json.JSONDecodeError, IOError, TypeError, KeyError) as e:
                print(f"Error loading graph: {e}. Creating a new graph.")
        
        return graph

    def save_graph(self):
        """
        Saves the knowledge graph to a JSON file.

        Raises:
            IOError: If there is an error writing to the file.
        """
        try:
            # Try different versions of NetworkX API
            try:
                data = nx.node_link_data(self.graph, edges="links")
            except TypeError:
                # Older NetworkX doesn't support the edges parameter
                data = nx.node_link_data(self.graph)
                # Rename 'edges' to 'links' if desired
                if 'edges' in data:
                    data['links'] = data.pop('edges')
                    
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except (IOError, TypeError) as e:
            print(f"Error saving graph: {e}")

    def node_exists(self, node):
        """
        Checks if a node exists in the graph.

        Args:
            node (str): The node to check.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        return node in self.graph.nodes

    def add_relationship(self, node1, node2, relationship):
        """
        Adds a relationship between two nodes in the knowledge graph.

        Args:
            node1 (str): The first node.
            node2 (str): The second node.
            relationship (str): The relationship between the nodes.

        Raises:
            ValueError: If any of the nodes or relationship is empty.
        """
        self._validate_non_empty(node1, node2, relationship)
        self.graph.add_edge(node1, node2, relationship=relationship)
        self._assert_node_exists(node1, node2)
        self.save_graph()

    def add_knowledge(self, node, relationship, related_node, metadata=None):
        """
        Adds new knowledge to the knowledge graph.

        Args:
            node (str): The source node.
            relationship (str): The relationship between the nodes.
            related_node (str): The target node.
            metadata (dict): Additional metadata for the relationship.

        Raises:
            ValueError: If any of the nodes or relationship is empty.
        """
        self._validate_non_empty(node, related_node, relationship)
        metadata = metadata or {}
        self.graph.add_edge(node, related_node, relationship=relationship, metadata=metadata)
        self.save_graph()

    def traverse_graph(self, query):
        """
        Finds related concepts using graph traversal based on a query.

        Args:
            query (str): The query string to search for in the graph nodes.

        Returns:
            list: A list of dictionaries containing related concepts and their relationships.

        Raises:
            ValueError: If the query is empty.
        """
        self._validate_non_empty(query)
        cache_key = f"graph_traversal:{query}"
        results = self.memory_manager.get_cache(cache_key)
        if results is None:
            results = self._search_graph(query)
            self.memory_manager.set_cache(cache_key, results, ttl=604800)  # Cache for 1 week
        return results

    def find_related_concepts(self, query, depth=2):
        """
        Finds related concepts using graph algorithms.

        Args:
            query (str): The query string to search for in the graph nodes.
            depth (int): The depth of the graph traversal.

        Returns:
            list: A list of related concepts.
        """
        self._validate_non_empty(query)
        related_concepts = set()
        for node in self.graph.nodes:
            if query.lower() in node.lower():
                paths = nx.single_source_shortest_path(self.graph, node, cutoff=depth)
                for path in paths.values():
                    related_concepts.update(path)
        return list(related_concepts)

    def _validate_non_empty(self, *args):
        """Validates that all arguments are non-empty."""
        if any(not arg for arg in args):
            raise ValueError("Arguments must be non-empty.")

    def _assert_node_exists(self, *nodes):
        """Asserts that all nodes exist in the graph."""
        for node in nodes:
            assert self.node_exists(node), f"Node {node} was not added to the graph."

    def _search_graph(self, query):
        """Searches the graph for nodes matching the query."""
        results = []
        for node in self.graph.nodes:
            if query.lower() in node.lower():
                neighbors = list(self.graph.neighbors(node))
                for neighbor in neighbors:
                    relationship = self.graph[node][neighbor]['relationship']
                    results.append({"concept": neighbor, "relationship": relationship})
        return results

    def add_node(self, node_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> GraphNode:
        """Add a node to the graph with vector embedding."""
        metadata = metadata or {}
        embedding = self.embeddings.embed_query(content)
        
        node = GraphNode(
            id=node_id,
            content=content,
            metadata=metadata,
            embedding=embedding
        )
        
        self.graph.add_node(node_id, **metadata)
        self.node_cache[node_id] = node
        
        # Add to vector store
        self.vector_store.add_texts(
            texts=[content],
            metadatas=[metadata],
            embeddings=[embedding]
        )
        
        return node

    def add_edge(self, source: str, target: str, weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> GraphEdge:
        """Add an edge to the graph."""
        if source not in self.node_cache or target not in self.node_cache:
            raise ValueError("Source and target nodes must exist")
            
        metadata = metadata or {}
        edge = GraphEdge(
            source=source,
            target=target,
            weight=weight,
            metadata=metadata
        )
        
        self.graph.add_edge(source, target, weight=weight, **metadata)
        self.edge_cache[(source, target)] = edge
        
        return edge

    def search(self, query: str, k: int = 5, search_type: str = "hybrid") -> List[Tuple[GraphNode, float]]:
        """Search for nodes using different search strategies."""
        if search_type == "graph":
            return self._graph_search(query, k)
        elif search_type == "semantic":
            return self._semantic_search(query, k)
        elif search_type == "hybrid":
            return self._hybrid_search(query, k)
        else:
            raise ValueError(f"Invalid search type: {search_type}")

    def _graph_search(self, query: str, k: int) -> List[Tuple[GraphNode, float]]:
        """Search using graph traversal."""
        # Find starting nodes using semantic search
        start_nodes = self._semantic_search(query, k=3)
        
        results = []
        for node, score in start_nodes:
            # Perform breadth-first search from each start node
            bfs_nodes = nx.bfs_tree(self.graph, node.id)
            for n_id in bfs_nodes:
                if n_id in self.node_cache:
                    results.append((self.node_cache[n_id], score))
        
        # Sort by score and take top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _semantic_search(self, query: str, k: int) -> List[Tuple[GraphNode, float]]:
        """Search using semantic similarity."""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            (self.node_cache[doc.metadata["node_id"]], score)
            for doc, score in results
            if doc.metadata["node_id"] in self.node_cache
        ]

    def _hybrid_search(self, query: str, k: int) -> List[Tuple[GraphNode, float]]:
        """Combine graph and semantic search."""
        graph_results = self._graph_search(query, k)
        semantic_results = self._semantic_search(query, k)
        
        # Combine and deduplicate results
        seen_nodes = set()
        results = []
        
        for node, score in graph_results + semantic_results:
            if node.id not in seen_nodes:
                seen_nodes.add(node.id)
                results.append((node, score))
        
        # Sort by score and take top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def find_path(self, source: str, target: str) -> List[GraphNode]:
        """Find the shortest path between two nodes."""
        try:
            path = nx.shortest_path(self.graph, source, target)
            return [self.node_cache[node_id] for node_id in path]
        except nx.NetworkXNoPath:
            return []

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[GraphNode]:
        """Get neighboring nodes in specified direction."""
        if direction == "in":
            neighbors = self.graph.predecessors(node_id)
        elif direction == "out":
            neighbors = self.graph.successors(node_id)
        elif direction == "both":
            neighbors = set(self.graph.predecessors(node_id)) | set(self.graph.successors(node_id))
        else:
            raise ValueError(f"Invalid direction: {direction}")
            
        return [self.node_cache[n_id] for n_id in neighbors if n_id in self.node_cache]

    def get_edge_data(self, source: str, target: str) -> Optional[GraphEdge]:
        """Get edge data between two nodes."""
        return self.edge_cache.get((source, target))

    def remove_node(self, node_id: str):
        """Remove a node and its edges from the graph."""
        if node_id in self.node_cache:
            # Remove from vector store
            self.vector_store.delete([node_id])
            
            # Remove from graph
            self.graph.remove_node(node_id)
            
            # Remove from caches
            del self.node_cache[node_id]
            edges_to_remove = [
                (s, t) for s, t in self.edge_cache.keys()
                if s == node_id or t == node_id
            ]
            for edge in edges_to_remove:
                del self.edge_cache[edge]

    def remove_edge(self, source: str, target: str):
        """Remove an edge from the graph."""
        if (source, target) in self.edge_cache:
            self.graph.remove_edge(source, target)
            del self.edge_cache[(source, target)]

    def clear(self):
        """Clear the graph and all caches."""
        self.graph.clear()
        self.node_cache.clear()
        self.edge_cache.clear()
        self.vector_store.delete_all()

    def get_node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return len(self.node_cache)

    def get_edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return len(self.edge_cache)

    def get_connected_components(self) -> List[Set[str]]:
        """Get the connected components of the graph."""
        return list(nx.weakly_connected_components(self.graph))

    def get_node_degree(self, node_id: str) -> Dict[str, int]:
        """Get the in and out degree of a node."""
        return {
            "in": self.graph.in_degree(node_id),
            "out": self.graph.out_degree(node_id)
        }
