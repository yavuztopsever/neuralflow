import pytest
import asyncio
from datetime import datetime
from pathlib import Path
import json
import shutil

from neuralflow.core.workflow.base import WorkflowNode, WorkflowEdge
from neuralflow.core.workflow.implementations import SequentialWorkflow, ParallelWorkflow
from neuralflow.core.models.management.workflow_manager import WorkflowManager
from neuralflow.core.services.workflow_executor import WorkflowExecutionService

class TestNode(WorkflowNode):
    """Test node implementation."""
    
    async def execute(self) -> dict:
        await asyncio.sleep(0.1)  # Simulate work
        return {"result": f"Node {self.node_id} executed"}
    
    async def validate(self) -> bool:
        return True

@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary storage directory."""
    storage_path = tmp_path / "workflows"
    storage_path.mkdir()
    yield storage_path
    shutil.rmtree(storage_path)

@pytest.fixture
async def workflow_manager(temp_storage):
    """Create a workflow manager instance."""
    return WorkflowManager(str(temp_storage))

@pytest.fixture
async def workflow_executor(workflow_manager):
    """Create a workflow executor instance."""
    return WorkflowExecutionService(workflow_manager)

@pytest.mark.asyncio
async def test_sequential_workflow(workflow_manager, workflow_executor):
    """Test sequential workflow execution."""
    # Create workflow
    workflow = await workflow_manager.create_workflow("Test Sequential", "sequential")
    
    # Add nodes
    node1 = TestNode("node1", "Node 1")
    node2 = TestNode("node2", "Node 2")
    node3 = TestNode("node3", "Node 3")
    
    await workflow_manager.add_node(workflow.workflow_id, node1)
    await workflow_manager.add_node(workflow.workflow_id, node2)
    await workflow_manager.add_node(workflow.workflow_id, node3)
    
    # Add edges
    edge1 = WorkflowEdge("node1", "node2")
    edge2 = WorkflowEdge("node2", "node3")
    
    await workflow_manager.add_edge(workflow.workflow_id, edge1)
    await workflow_manager.add_edge(workflow.workflow_id, edge2)
    
    # Execute workflow
    results = await workflow_executor.execute_workflow(workflow.workflow_id)
    
    # Verify results
    assert len(results) == 3
    assert results["node1"]["result"] == "Node node1 executed"
    assert results["node2"]["result"] == "Node node2 executed"
    assert results["node3"]["result"] == "Node node3 executed"
    
    # Verify workflow status
    status = await workflow_executor.get_execution_status(workflow.workflow_id)
    assert status["status"] == "completed"
    assert len(status["nodes"]) == 3
    assert all(node["status"] == "completed" for node in status["nodes"])

@pytest.mark.asyncio
async def test_parallel_workflow(workflow_manager, workflow_executor):
    """Test parallel workflow execution."""
    # Create workflow
    workflow = await workflow_manager.create_workflow("Test Parallel", "parallel")
    
    # Add nodes
    node1 = TestNode("node1", "Node 1")
    node2 = TestNode("node2", "Node 2")
    node3 = TestNode("node3", "Node 3")
    
    await workflow_manager.add_node(workflow.workflow_id, node1)
    await workflow_manager.add_node(workflow.workflow_id, node2)
    await workflow_manager.add_node(workflow.workflow_id, node3)
    
    # Add edges (all nodes can run in parallel)
    edge1 = WorkflowEdge("node1", "node2")
    edge2 = WorkflowEdge("node2", "node3")
    
    await workflow_manager.add_edge(workflow.workflow_id, edge1)
    await workflow_manager.add_edge(workflow.workflow_id, edge2)
    
    # Execute workflow
    start_time = datetime.now()
    results = await workflow_executor.execute_workflow(workflow.workflow_id)
    end_time = datetime.now()
    
    # Verify results
    assert len(results) == 3
    assert results["node1"]["result"] == "Node node1 executed"
    assert results["node2"]["result"] == "Node node2 executed"
    assert results["node3"]["result"] == "Node node3 executed"
    
    # Verify parallel execution (should take less than 0.3 seconds)
    execution_time = (end_time - start_time).total_seconds()
    assert execution_time < 0.3  # All nodes run in parallel (0.1s each)
    
    # Verify workflow status
    status = await workflow_executor.get_execution_status(workflow.workflow_id)
    assert status["status"] == "completed"
    assert len(status["nodes"]) == 3
    assert all(node["status"] == "completed" for node in status["nodes"])

@pytest.mark.asyncio
async def test_workflow_persistence(workflow_manager, temp_storage):
    """Test workflow persistence."""
    # Create workflow
    workflow = await workflow_manager.create_workflow("Test Persistence", "sequential")
    
    # Add node
    node = TestNode("node1", "Node 1")
    await workflow_manager.add_node(workflow.workflow_id, node)
    
    # Verify workflow is saved
    workflow_file = temp_storage / f"{workflow.workflow_id}.json"
    assert workflow_file.exists()
    
    # Load workflow from file
    with open(workflow_file) as f:
        saved_data = json.load(f)
    
    assert saved_data["workflow_id"] == workflow.workflow_id
    assert saved_data["name"] == "Test Persistence"
    assert saved_data["type"] == "sequential"
    assert len(saved_data["nodes"]) == 1
    assert saved_data["nodes"][0]["node_id"] == "node1"

@pytest.mark.asyncio
async def test_workflow_cancellation(workflow_manager, workflow_executor):
    """Test workflow cancellation."""
    # Create workflow
    workflow = await workflow_manager.create_workflow("Test Cancellation", "sequential")
    
    # Add node with long execution time
    class SlowNode(WorkflowNode):
        async def execute(self) -> dict:
            await asyncio.sleep(1)  # Long execution time
            return {"result": "Node executed"}
        
        async def validate(self) -> bool:
            return True
    
    node = SlowNode("node1", "Slow Node")
    await workflow_manager.add_node(workflow.workflow_id, node)
    
    # Start workflow execution
    execution_task = asyncio.create_task(workflow_executor.execute_workflow(workflow.workflow_id))
    
    # Wait a bit and cancel
    await asyncio.sleep(0.1)
    success = await workflow_executor.cancel_workflow(workflow.workflow_id)
    
    assert success
    
    # Verify workflow was cancelled
    with pytest.raises(asyncio.CancelledError):
        await execution_task
    
    # Verify workflow status
    status = await workflow_executor.get_execution_status(workflow.workflow_id)
    assert status["status"] == "cancelled"
    assert status["nodes"][0]["status"] == "pending"

@pytest.mark.asyncio
async def test_execution_history(workflow_manager, workflow_executor):
    """Test execution history tracking."""
    # Create workflow
    workflow = await workflow_manager.create_workflow("Test History", "sequential")
    
    # Add node
    node = TestNode("node1", "Node 1")
    await workflow_manager.add_node(workflow.workflow_id, node)
    
    # Execute workflow multiple times
    for _ in range(3):
        await workflow_executor.execute_workflow(workflow.workflow_id)
    
    # Verify execution history
    status = await workflow_executor.get_execution_status(workflow.workflow_id)
    assert "execution_history" in status
    assert len(status["execution_history"]) == 3
    
    # Verify history entries
    for entry in status["execution_history"]:
        assert "execution_id" in entry
        assert "start_time" in entry
        assert "end_time" in entry
        assert entry["status"] == "completed"
        assert "results" in entry
    
    # Test history cleanup
    await workflow_executor.cleanup_old_executions(max_age_days=0)
    status = await workflow_executor.get_execution_status(workflow.workflow_id)
    assert len(status["execution_history"]) == 0 