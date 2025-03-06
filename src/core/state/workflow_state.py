"""
Workflow state representation and management.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class WorkflowState:
    """Represents the state of a workflow execution."""
    
    workflow_id: str
    state_id: str
    context: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update(self, results: Dict[str, Any],
               status: str = "completed",
               error: Optional[str] = None) -> None:
        """Update the state with execution results."""
        self.results = results
        self.status = status
        self.error = error
        self.modified = datetime.now().isoformat()
        
        # Add to history
        self.history.append({
            "timestamp": self.modified,
            "status": status,
            "results": results,
            "error": error
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "state_id": self.state_id,
            "context": self.context,
            "metadata": self.metadata,
            "created": self.created,
            "modified": self.modified,
            "status": self.status,
            "results": self.results,
            "error": self.error,
            "history": self.history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        """Create state from dictionary."""
        return cls(
            workflow_id=data["workflow_id"],
            state_id=data["state_id"],
            context=data["context"],
            metadata=data.get("metadata", {}),
            created=data["created"],
            modified=data["modified"],
            status=data["status"],
            results=data["results"],
            error=data.get("error"),
            history=data.get("history", [])
        )
    
    def is_completed(self) -> bool:
        """Check if the state is completed."""
        return self.status == "completed"
    
    def is_failed(self) -> bool:
        """Check if the state has failed."""
        return self.status == "failed"
    
    def is_pending(self) -> bool:
        """Check if the state is pending."""
        return self.status == "pending"
    
    def get_latest_history_entry(self) -> Optional[Dict[str, Any]]:
        """Get the latest history entry."""
        return self.history[-1] if self.history else None 