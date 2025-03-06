import os
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Union
import aiohttp
import bs4
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from config.config import Config

class RAGManager:
    """Manages RAG operations and web crawling capabilities."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=str(config.VECTOR_DB_DIR),
            anonymized_telemetry=False
        ))
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDER_MODEL
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function
        )
        
        # Initialize web crawling settings
        self.web_crawl_timeout = 30
        self.max_web_pages = 3
        self.web_crawl_depth = 2
        
    async def crawl_web(self, query: str) -> List[Dict]:
        """Crawl the web for relevant information."""
        try:
            # Create search query
            search_url = f"https://www.google.com/search?q={query}"
            
            async with aiohttp.ClientSession() as session:
                # Get search results
                async with session.get(search_url, timeout=self.web_crawl_timeout) as response:
                    if response.status != 200:
                        return []
                        
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract search results
                    results = []
                    for result in soup.select('div.g')[:self.max_web_pages]:
                        try:
                            title_elem = result.select_one('h3')
                            link_elem = result.select_one('a')
                            snippet_elem = result.select_one('div.VwiC3b')
                            
                            if title_elem and link_elem and snippet_elem:
                                results.append({
                                    'title': title_elem.text,
                                    'url': link_elem['href'],
                                    'snippet': snippet_elem.text
                                })
                        except Exception as e:
                            self.logger.error(f"Error parsing search result: {e}")
                            continue
                            
                    # Crawl each result
                    crawled_data = []
                    for result in results:
                        try:
                            content = await self._crawl_page(result['url'])
                            if content:
                                crawled_data.append({
                                    'title': result['title'],
                                    'url': result['url'],
                                    'content': content
                                })
                        except Exception as e:
                            self.logger.error(f"Error crawling {result['url']}: {e}")
                            continue
                            
                    return crawled_data
                    
        except Exception as e:
            self.logger.error(f"Error in web crawling: {e}")
            return []
            
    async def _crawl_page(self, url: str) -> Optional[str]:
        """Crawl a single page and extract relevant content."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.web_crawl_timeout) as response:
                    if response.status != 200:
                        return None
                        
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                        element.decompose()
                        
                    # Extract main content
                    content = soup.get_text()
                    
                    # Clean up content
                    content = ' '.join(content.split())
                    
                    return content
                    
        except Exception as e:
            self.logger.error(f"Error crawling page {url}: {e}")
            return None
            
    def add_documents(self, documents: List[Dict[str, Union[str, Dict]]]):
        """Add documents to the vector store."""
        try:
            # Prepare documents for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                # Generate unique ID
                doc_id = f"doc_{len(ids)}"
                ids.append(doc_id)
                
                # Prepare text
                if isinstance(doc['content'], str):
                    text = doc['content']
                else:
                    text = str(doc['content'])
                texts.append(text)
                
                # Prepare metadata
                metadata = {
                    'title': doc.get('title', ''),
                    'url': doc.get('url', ''),
                    'source': doc.get('source', 'local'),
                    'timestamp': doc.get('timestamp', '')
                }
                if 'metadata' in doc:
                    metadata.update(doc['metadata'])
                metadatas.append(metadata)
                
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            
    def query_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the vector store for relevant documents."""
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
                
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error querying documents: {e}")
            return []
            
    def get_context(self, query: str, max_results: int = 5) -> Dict:
        """Get context from both local documents and web crawling."""
        try:
            # Get local document results
            local_results = self.query_documents(query, n_results=max_results)
            
            # Get web results
            web_results = asyncio.run(self.crawl_web(query))
            
            # Combine and sort results
            all_results = []
            
            # Add local results with higher priority
            for result in local_results:
                all_results.append({
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'source': 'local',
                    'relevance': 1 - (result['distance'] if result['distance'] is not None else 0)
                })
                
            # Add web results with lower priority
            for result in web_results:
                all_results.append({
                    'content': result['content'],
                    'metadata': {
                        'title': result['title'],
                        'url': result['url']
                    },
                    'source': 'web',
                    'relevance': 0.7  # Lower priority for web results
                })
                
            # Sort by relevance
            all_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            # Take top results
            return {
                'local_results': local_results,
                'web_results': web_results,
                'combined_results': all_results[:max_results]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting context: {e}")
            return {
                'local_results': [],
                'web_results': [],
                'combined_results': []
            }
            
    def clear_collection(self):
        """Clear the vector store collection."""
        try:
            self.collection.delete()
            self.logger.info("Cleared vector store collection")
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
            
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store collection."""
        try:
            return {
                'count': self.collection.count(),
                'name': self.collection.name,
                'metadata': self.collection.get()
            }
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {} 