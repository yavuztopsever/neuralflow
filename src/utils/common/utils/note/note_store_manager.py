import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import re
from datetime import datetime
import json
from config.config import Config

class ObsidianIntegration:
    """Handles integration with Obsidian vaults for knowledge infusion."""
    
    def __init__(self, vault_path: str, config: Config):
        self.vault_path = Path(vault_path)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validate vault path
        if not self.vault_path.exists():
            raise ValueError(f"Obsidian vault not found at {vault_path}")
            
        # Initialize metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load or create metadata for the vault."""
        metadata_path = self.vault_path / ".obsidian" / "metadata.json"
        try:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            return {"last_processed": None, "processed_files": {}}
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            return {"last_processed": None, "processed_files": {}}
            
    def _save_metadata(self):
        """Save metadata to the vault."""
        metadata_path = self.vault_path / ".obsidian" / "metadata.json"
        try:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
            
    def get_markdown_files(self) -> List[Path]:
        """Get all markdown files from the vault."""
        try:
            return list(self.vault_path.rglob("*.md"))
        except Exception as e:
            self.logger.error(f"Error getting markdown files: {e}")
            return []
            
    def extract_metadata(self, content: str) -> Dict:
        """Extract metadata from markdown content."""
        metadata = {}
        
        # Extract frontmatter if present
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            for line in frontmatter.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
                    
        # Extract tags
        tags = re.findall(r'#(\w+)', content)
        if tags:
            metadata['tags'] = list(set(tags))
            
        # Extract links
        links = re.findall(r'\[\[(.*?)\]\]', content)
        if links:
            metadata['links'] = list(set(links))
            
        return metadata
        
    def process_file(self, file_path: Path) -> Optional[Dict]:
        """Process a single markdown file."""
        try:
            # Check if file was already processed
            file_id = str(file_path.relative_to(self.vault_path))
            if file_id in self.metadata["processed_files"]:
                last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                last_processed = datetime.fromisoformat(self.metadata["processed_files"][file_id]["last_processed"])
                if last_modified <= last_processed:
                    return None
                    
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract metadata
            metadata = self.extract_metadata(content)
            
            # Clean content
            cleaned_content = self._clean_content(content)
            
            # Create document
            document = {
                "id": file_id,
                "title": file_path.stem,
                "content": cleaned_content,
                "metadata": metadata,
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "processed_at": datetime.now().isoformat()
            }
            
            # Update metadata
            self.metadata["processed_files"][file_id] = {
                "last_processed": document["processed_at"],
                "title": document["title"]
            }
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return None
            
    def _clean_content(self, content: str) -> str:
        """Clean markdown content for training."""
        # Remove frontmatter
        content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
        
        # Remove code blocks
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        
        # Remove inline code
        content = re.sub(r'`.*?`', '', content)
        
        # Remove links but keep link text
        content = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', content)
        
        # Remove wiki-style links but keep link text
        content = re.sub(r'\[\[(.*?)\]\]', r'\1', content)
        
        # Remove images
        content = re.sub(r'!\[.*?\]\((.*?)\)', '', content)
        
        # Remove HTML tags
        content = re.sub(r'<.*?>', '', content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
        
    def get_training_data(self) -> List[Dict]:
        """Get training data from the vault."""
        training_data = []
        
        try:
            # Get all markdown files
            files = self.get_markdown_files()
            
            # Process each file
            for file_path in files:
                document = self.process_file(file_path)
                if document:
                    training_data.append(document)
                    
            # Update last processed timestamp
            self.metadata["last_processed"] = datetime.now().isoformat()
            self._save_metadata()
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {e}")
            return []
            
    def save_training_data(self, training_data: List[Dict], output_dir: Path):
        """Save processed training data to disk."""
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as JSONL file
            output_file = output_dir / "obsidian_training_data.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for document in training_data:
                    f.write(json.dumps(document) + '\n')
                    
            self.logger.info(f"Saved {len(training_data)} documents to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving training data: {e}")
            
    def get_vault_summary(self) -> Dict:
        """Get a summary of the vault's content."""
        try:
            files = self.get_markdown_files()
            total_files = len(files)
            processed_files = len(self.metadata["processed_files"])
            
            # Get file size statistics
            sizes = [f.stat().st_size for f in files]
            avg_size = sum(sizes) / len(sizes) if sizes else 0
            
            # Get tag statistics
            all_tags = set()
            for file_id, info in self.metadata["processed_files"].items():
                if "tags" in info:
                    all_tags.update(info["tags"])
                    
            return {
                "total_files": total_files,
                "processed_files": processed_files,
                "average_file_size": avg_size,
                "total_tags": len(all_tags),
                "last_processed": self.metadata["last_processed"]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting vault summary: {e}")
            return {} 