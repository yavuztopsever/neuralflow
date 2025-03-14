"""
Unit tests for workflow visualization component functionality.
"""
import pytest
from datetime import datetime
from typing import Dict, Any, List

from src.ui.workflow import WorkflowVisualizer, Node, Edge, WorkflowGraph

class TestWorkflowVisualizer:
    """Test suite for workflow visualization functionality."""
    
    @pytest.fixture
    def workflow_visualizer(self):
        """Create a workflow visualizer for testing."""
        return WorkflowVisualizer(
            theme="light",
            layout="hierarchical",
            auto_layout=True
        )
    
    @pytest.fixture
    def sample_workflow(self):
        """Create a sample workflow for testing."""
        nodes = [
            Node(
                id="input",
                type="input",
                label="Input Node",
                position={"x": 0, "y": 0},
                data={"config": {}}
            ),
            Node(
                id="process",
                type="process",
                label="Process Node",
                position={"x": 100, "y": 0},
                data={"config": {"timeout": 30}}
            ),
            Node(
                id="output",
                type="output",
                label="Output Node",
                position={"x": 200, "y": 0},
                data={"config": {}}
            )
        ]
        
        edges = [
            Edge(
                id="edge1",
                source="input",
                target="process",
                label="Process",
                data={"type": "data"}
            ),
            Edge(
                id="edge2",
                source="process",
                target="output",
                label="Output",
                data={"type": "result"}
            )
        ]
        
        return WorkflowGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                "name": "Test Workflow",
                "description": "Test workflow for visualization",
                "created_at": datetime.now()
            }
        )
    
    @pytest.mark.asyncio
    async def test_visualizer_initialization(self, workflow_visualizer):
        """Test workflow visualizer initialization."""
        assert workflow_visualizer.theme == "light"
        assert workflow_visualizer.layout == "hierarchical"
        assert workflow_visualizer.auto_layout is True
        assert workflow_visualizer._graphs == {}
        assert workflow_visualizer._layouts == {}
    
    @pytest.mark.asyncio
    async def test_workflow_management(self, workflow_visualizer, sample_workflow):
        """Test workflow management operations."""
        workflow_id = "test_workflow"
        
        # Add workflow
        await workflow_visualizer.add_workflow(workflow_id, sample_workflow)
        assert workflow_id in workflow_visualizer._graphs
        stored_workflow = workflow_visualizer._graphs[workflow_id]
        assert len(stored_workflow.nodes) == len(sample_workflow.nodes)
        assert len(stored_workflow.edges) == len(sample_workflow.edges)
        
        # Get workflow
        retrieved_workflow = await workflow_visualizer.get_workflow(workflow_id)
        assert retrieved_workflow is not None
        assert retrieved_workflow.metadata["name"] == sample_workflow.metadata["name"]
        
        # Update workflow
        updated_workflow = WorkflowGraph(
            nodes=sample_workflow.nodes,
            edges=sample_workflow.edges,
            metadata={"name": "Updated Workflow"}
        )
        await workflow_visualizer.update_workflow(workflow_id, updated_workflow)
        retrieved_workflow = await workflow_visualizer.get_workflow(workflow_id)
        assert retrieved_workflow.metadata["name"] == "Updated Workflow"
        
        # Delete workflow
        await workflow_visualizer.delete_workflow(workflow_id)
        assert workflow_id not in workflow_visualizer._graphs
    
    @pytest.mark.asyncio
    async def test_node_management(self, workflow_visualizer, sample_workflow):
        """Test node management operations."""
        workflow_id = "test_workflow"
        await workflow_visualizer.add_workflow(workflow_id, sample_workflow)
        
        # Add node
        new_node = Node(
            id="new_node",
            type="process",
            label="New Node",
            position={"x": 150, "y": 100},
            data={"config": {}}
        )
        await workflow_visualizer.add_node(workflow_id, new_node)
        workflow = await workflow_visualizer.get_workflow(workflow_id)
        assert len(workflow.nodes) == len(sample_workflow.nodes) + 1
        
        # Update node
        updated_node = Node(
            id="new_node",
            type="process",
            label="Updated Node",
            position={"x": 150, "y": 100},
            data={"config": {}}
        )
        await workflow_visualizer.update_node(workflow_id, updated_node)
        workflow = await workflow_visualizer.get_workflow(workflow_id)
        node = next(n for n in workflow.nodes if n.id == "new_node")
        assert node.label == "Updated Node"
        
        # Delete node
        await workflow_visualizer.delete_node(workflow_id, "new_node")
        workflow = await workflow_visualizer.get_workflow(workflow_id)
        assert len(workflow.nodes) == len(sample_workflow.nodes)
    
    @pytest.mark.asyncio
    async def test_edge_management(self, workflow_visualizer, sample_workflow):
        """Test edge management operations."""
        workflow_id = "test_workflow"
        await workflow_visualizer.add_workflow(workflow_id, sample_workflow)
        
        # Add edge
        new_edge = Edge(
            id="edge3",
            source="process",
            target="output",
            label="New Edge",
            data={"type": "data"}
        )
        await workflow_visualizer.add_edge(workflow_id, new_edge)
        workflow = await workflow_visualizer.get_workflow(workflow_id)
        assert len(workflow.edges) == len(sample_workflow.edges) + 1
        
        # Update edge
        updated_edge = Edge(
            id="edge3",
            source="process",
            target="output",
            label="Updated Edge",
            data={"type": "data"}
        )
        await workflow_visualizer.update_edge(workflow_id, updated_edge)
        workflow = await workflow_visualizer.get_workflow(workflow_id)
        edge = next(e for e in workflow.edges if e.id == "edge3")
        assert edge.label == "Updated Edge"
        
        # Delete edge
        await workflow_visualizer.delete_edge(workflow_id, "edge3")
        workflow = await workflow_visualizer.get_workflow(workflow_id)
        assert len(workflow.edges) == len(sample_workflow.edges)
    
    @pytest.mark.asyncio
    async def test_layout_management(self, workflow_visualizer, sample_workflow):
        """Test layout management operations."""
        workflow_id = "test_workflow"
        await workflow_visualizer.add_workflow(workflow_id, sample_workflow)
        
        # Test different layouts
        layouts = ["hierarchical", "force-directed", "circular", "grid"]
        for layout in layouts:
            workflow_visualizer.layout = layout
            await workflow_visualizer.apply_layout(workflow_id)
            assert workflow_id in workflow_visualizer._layouts
            assert workflow_visualizer._layouts[workflow_id]["type"] == layout
    
    @pytest.mark.asyncio
    async def test_theme_management(self, workflow_visualizer):
        """Test theme management operations."""
        # Test theme switching
        themes = ["light", "dark", "custom"]
        for theme in themes:
            workflow_visualizer.theme = theme
            await workflow_visualizer.apply_theme()
            assert workflow_visualizer._current_theme == theme
        
        # Test custom theme
        custom_theme = {
            "background": "#ffffff",
            "node_color": "#000000",
            "edge_color": "#666666"
        }
        workflow_visualizer.apply_custom_theme(custom_theme)
        assert workflow_visualizer._current_theme == "custom"
        assert workflow_visualizer._theme_colors == custom_theme
    
    @pytest.mark.asyncio
    async def test_workflow_metrics(self, workflow_visualizer, sample_workflow):
        """Test workflow metrics collection."""
        workflow_id = "test_workflow"
        await workflow_visualizer.add_workflow(workflow_id, sample_workflow)
        
        # Get metrics
        metrics = await workflow_visualizer.get_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "total_workflows" in metrics
        assert "total_nodes" in metrics
        assert "total_edges" in metrics
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, workflow_visualizer):
        """Test workflow error handling."""
        # Test invalid workflow ID
        with pytest.raises(ValueError):
            await workflow_visualizer.add_workflow(None, None)
        
        # Test invalid workflow data
        with pytest.raises(ValueError):
            await workflow_visualizer.add_workflow("test_workflow", None)
        
        # Test getting non-existent workflow
        workflow = await workflow_visualizer.get_workflow("non_existent")
        assert workflow is None
        
        # Test updating non-existent workflow
        with pytest.raises(KeyError):
            await workflow_visualizer.update_workflow("non_existent", None)
        
        # Test deleting non-existent workflow
        with pytest.raises(KeyError):
            await workflow_visualizer.delete_workflow("non_existent")
        
        # Test invalid layout
        with pytest.raises(ValueError):
            workflow_visualizer.layout = "invalid_layout"
            await workflow_visualizer.apply_layout("test_workflow")
        
        # Test invalid theme
        with pytest.raises(ValueError):
            workflow_visualizer.theme = "invalid_theme"
            await workflow_visualizer.apply_theme() 