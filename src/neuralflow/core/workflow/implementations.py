from typing import Any, Dict, List, Optional
from .base import Workflow, WorkflowNode, WorkflowEdge

class SequentialWorkflow(Workflow):
    """A workflow that executes nodes in a sequential order."""
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the workflow nodes in sequence."""
        results = {}
        self.status = "running"
        
        try:
            # Get nodes in execution order (topological sort)
            execution_order = self._get_execution_order()
            
            for node_id in execution_order:
                node = self.nodes[node_id]
                node.status = "running"
                
                # Execute node
                node_results = await node.execute()
                results[node_id] = node_results
                
                # Update node status
                node.status = "completed"
                node.outputs = node_results
                
            self.status = "completed"
            return results
            
        except Exception as e:
            self.status = "failed"
            raise RuntimeError(f"Workflow execution failed: {str(e)}")
    
    async def validate(self) -> bool:
        """Validate the workflow configuration."""
        # Check for cycles
        if self._has_cycles():
            return False
        
        # Validate all nodes
        for node in self.nodes.values():
            if not await node.validate():
                return False
        
        return True
    
    def _get_execution_order(self) -> List[str]:
        """Get the execution order of nodes using topological sort."""
        visited = set()
        temp = set()
        order = []
        
        def visit(node_id: str) -> None:
            if node_id in temp:
                raise ValueError("Cycle detected in workflow")
            if node_id not in visited:
                temp.add(node_id)
                for dep_id in self.get_node_dependencies(node_id):
                    visit(dep_id)
                temp.remove(node_id)
                visited.add(node_id)
                order.append(node_id)
        
        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)
        
        return order
    
    def _has_cycles(self) -> bool:
        """Check if the workflow has cycles."""
        visited = set()
        temp = set()
        
        def has_cycle(node_id: str) -> bool:
            if node_id in temp:
                return True
            if node_id in visited:
                return False
            
            temp.add(node_id)
            visited.add(node_id)
            
            for dep_id in self.get_node_dependencies(node_id):
                if has_cycle(dep_id):
                    return True
            
            temp.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if has_cycle(node_id):
                return True
        
        return False

class ParallelWorkflow(Workflow):
    """A workflow that can execute independent nodes in parallel."""
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the workflow nodes in parallel where possible."""
        results = {}
        self.status = "running"
        
        try:
            # Get execution groups (nodes that can run in parallel)
            execution_groups = self._get_execution_groups()
            
            for group in execution_groups:
                # Execute nodes in the group in parallel
                group_results = await self._execute_group(group)
                results.update(group_results)
            
            self.status = "completed"
            return results
            
        except Exception as e:
            self.status = "failed"
            raise RuntimeError(f"Workflow execution failed: {str(e)}")
    
    async def validate(self) -> bool:
        """Validate the workflow configuration."""
        # Check for cycles
        if self._has_cycles():
            return False
        
        # Validate all nodes
        for node in self.nodes.values():
            if not await node.validate():
                return False
        
        return True
    
    def _get_execution_groups(self) -> List[List[str]]:
        """Get groups of nodes that can be executed in parallel."""
        visited = set()
        groups = []
        
        def get_group(node_id: str, current_group: List[str]) -> None:
            if node_id in visited:
                return
            
            visited.add(node_id)
            current_group.append(node_id)
            
            # Get all nodes that can run in parallel with this node
            for dep_id in self.get_node_dependencies(node_id):
                get_group(dep_id, current_group)
        
        for node_id in self.nodes:
            if node_id not in visited:
                current_group = []
                get_group(node_id, current_group)
                if current_group:
                    groups.append(current_group)
        
        return groups
    
    async def _execute_group(self, group: List[str]) -> Dict[str, Any]:
        """Execute a group of nodes in parallel."""
        import asyncio
        
        async def execute_node(node_id: str) -> tuple[str, Dict[str, Any]]:
            node = self.nodes[node_id]
            node.status = "running"
            results = await node.execute()
            node.status = "completed"
            node.outputs = results
            return node_id, results
        
        # Execute all nodes in the group concurrently
        tasks = [execute_node(node_id) for node_id in group]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    def _has_cycles(self) -> bool:
        """Check if the workflow has cycles."""
        visited = set()
        temp = set()
        
        def has_cycle(node_id: str) -> bool:
            if node_id in temp:
                return True
            if node_id in visited:
                return False
            
            temp.add(node_id)
            visited.add(node_id)
            
            for dep_id in self.get_node_dependencies(node_id):
                if has_cycle(dep_id):
                    return True
            
            temp.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if has_cycle(node_id):
                return True
        
        return False 