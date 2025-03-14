"""
Unified workflow system for NeuralFlow.
"""

from typing import Dict, Any, Optional, List, Union, Type, Callable
from datetime import datetime
import logging
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
import json

from ..models.pipeline.unified_pipeline import UnifiedPipeline, PipelineStage, PipelineResult
from ..models.orchestration.unified_orchestrator import UnifiedOrchestrator
from ..models.experiments.unified_experiment import UnifiedExperiment
from ...utils.error.base_handler import BaseErrorHandler
from ...utils.logging.base_manager import BaseLogManager
from ...monitoring.core.unified_monitor import UnifiedMonitor, MonitoringConfig

logger = logging.getLogger(__name__)

class WorkflowConfig(BaseModel):
    """Workflow configuration."""
    storage_path: str = "storage/workflows"
    max_concurrent_pipelines: int = 5
    max_retries: int = 3
    retry_delay: float = 60.0  # seconds
    monitor_enabled: bool = True
    cache_enabled: bool = True
    timeout: float = 3600.0  # 1 hour

class WorkflowStage(BaseModel):
    """Workflow stage configuration."""
    stage_id: str
    pipeline_stages: List[PipelineStage]
    dependencies: List[str] = Field(default_factory=list)
    retry_policy: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class WorkflowResult(BaseModel):
    """Workflow execution result."""
    workflow_id: str
    stages: List[Dict[str, Any]]
    pipelines: Dict[str, PipelineResult]
    artifacts: Dict[str, str]
    metrics: Dict[str, float]
    status: str  # running, completed, failed
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class UnifiedWorkflow:
    """Unified workflow system for model development."""
    
    def __init__(
        self,
        orchestrator: UnifiedOrchestrator,
        experiment: Optional[UnifiedExperiment] = None,
        config: Optional[WorkflowConfig] = None,
        error_handler: Optional[BaseErrorHandler] = None,
        log_manager: Optional[BaseLogManager] = None
    ):
        """Initialize unified workflow.
        
        Args:
            orchestrator: Model orchestrator
            experiment: Optional experiment system
            config: Optional workflow configuration
            error_handler: Optional error handler
            log_manager: Optional log manager
        """
        self.orchestrator = orchestrator
        self.experiment = experiment
        self.config = config or WorkflowConfig()
        self.error_handler = error_handler or BaseErrorHandler()
        self.log_manager = log_manager or BaseLogManager()
        self.monitor = UnifiedMonitor(MonitoringConfig(
            monitor_id="model_workflow",
            monitor_type="workflow_system"
        )) if self.config.monitor_enabled else None
        
        # Initialize pipeline system
        self.pipeline = UnifiedPipeline(
            orchestrator=orchestrator,
            experiment=experiment,
            error_handler=error_handler,
            log_manager=log_manager
        )
        
        # Create workflow directory
        self.workflow_dir = Path(self.config.storage_path)
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize workflow storage
        self.workflows: Dict[str, WorkflowResult] = {}
        self.active_stages: Dict[str, asyncio.Task] = {}
        self.pipeline_semaphore = asyncio.Semaphore(self.config.max_concurrent_pipelines)
    
    async def run_workflow(
        self,
        workflow_id: str,
        stages: List[WorkflowStage],
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """Run a model workflow.
        
        Args:
            workflow_id: Workflow identifier
            stages: List of workflow stages
            metadata: Optional workflow metadata
            
        Returns:
            Workflow results
        """
        try:
            # Create workflow result
            workflow = WorkflowResult(
                workflow_id=workflow_id,
                stages=[],
                pipelines={},
                artifacts={},
                metrics={},
                status="running",
                metadata=metadata
            )
            self.workflows[workflow_id] = workflow
            
            # Build dependency graph
            graph = self._build_dependency_graph(stages)
            
            # Execute stages
            for stage_batch in graph:
                # Run stages in parallel
                tasks = []
                for stage in stage_batch:
                    task = asyncio.create_task(
                        self._run_stage(workflow, stage)
                    )
                    self.active_stages[stage.stage_id] = task
                    tasks.append(task)
                
                # Wait for batch completion
                try:
                    await asyncio.gather(*tasks)
                except Exception as e:
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    raise
                finally:
                    # Cleanup completed tasks
                    for stage in stage_batch:
                        if stage.stage_id in self.active_stages:
                            del self.active_stages[stage.stage_id]
            
            # Complete workflow
            workflow.status = "completed"
            workflow.end_time = datetime.now()
            
            # Save workflow
            await self._save_workflow(workflow)
            
            if self.monitor:
                await self.monitor.record_event({
                    "type": "workflow_completed",
                    "workflow_id": workflow_id,
                    "num_stages": len(stages),
                    "timestamp": datetime.now().isoformat()
                })
            
            return workflow
            
        except Exception as e:
            workflow.status = "failed"
            workflow.error = str(e)
            workflow.end_time = datetime.now()
            
            if self.monitor:
                await self.monitor.record_event({
                    "type": "workflow_failed",
                    "workflow_id": workflow_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            self.error_handler.handle_error(
                "WORKFLOW_ERROR",
                f"Failed to run workflow: {e}",
                details={"workflow_id": workflow_id}
            )
            raise
    
    async def get_workflow(
        self,
        workflow_id: str
    ) -> Optional[WorkflowResult]:
        """Get workflow results.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Workflow results if found
        """
        try:
            # Check memory
            if workflow_id in self.workflows:
                return self.workflows[workflow_id]
            
            # Check storage
            workflow_file = self.workflow_dir / f"{workflow_id}.json"
            if workflow_file.exists():
                with open(workflow_file) as f:
                    data = json.load(f)
                return WorkflowResult(**data)
            
            return None
            
        except Exception as e:
            self.error_handler.handle_error(
                "WORKFLOW_ERROR",
                f"Failed to get workflow: {e}",
                details={"workflow_id": workflow_id}
            )
            raise
    
    async def _run_stage(
        self,
        workflow: WorkflowResult,
        stage: WorkflowStage
    ) -> None:
        """Run a workflow stage.
        
        Args:
            workflow: Workflow result
            stage: Stage configuration
        """
        try:
            stage_result = {
                "stage_id": stage.stage_id,
                "status": "running",
                "start_time": datetime.now().isoformat()
            }
            workflow.stages.append(stage_result)
            
            # Run pipeline with retries
            retry_count = 0
            while True:
                try:
                    async with self.pipeline_semaphore:
                        pipeline = await self.pipeline.run_pipeline(
                            pipeline_id=f"{workflow.workflow_id}_{stage.stage_id}",
                            stages=stage.pipeline_stages,
                            metadata=stage.metadata
                        )
                    
                    # Store pipeline result
                    workflow.pipelines[stage.stage_id] = pipeline
                    
                    # Update artifacts and metrics
                    workflow.artifacts.update({
                        f"{stage.stage_id}_{k}": v
                        for k, v in pipeline.artifacts.items()
                    })
                    workflow.metrics.update(pipeline.metrics)
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if (
                        not stage.retry_policy or
                        retry_count >= (stage.retry_policy.get("max_retries") or self.config.max_retries)
                    ):
                        raise
                    
                    # Wait before retry
                    delay = stage.retry_policy.get("retry_delay") or self.config.retry_delay
                    await asyncio.sleep(delay)
            
            # Complete stage
            stage_result["status"] = "completed"
            stage_result["end_time"] = datetime.now().isoformat()
            
            if self.monitor:
                await self.monitor.record_event({
                    "type": "stage_completed",
                    "workflow_id": workflow.workflow_id,
                    "stage_id": stage.stage_id,
                    "timestamp": datetime.now().isoformat()
                })
            
        except Exception as e:
            stage_result["status"] = "failed"
            stage_result["error"] = str(e)
            stage_result["end_time"] = datetime.now().isoformat()
            
            if self.monitor:
                await self.monitor.record_event({
                    "type": "stage_failed",
                    "workflow_id": workflow.workflow_id,
                    "stage_id": stage.stage_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            raise
    
    def _build_dependency_graph(
        self,
        stages: List[WorkflowStage]
    ) -> List[List[WorkflowStage]]:
        """Build stage dependency graph.
        
        Args:
            stages: List of workflow stages
            
        Returns:
            List of stage batches that can be run in parallel
        """
        # Build dependency map
        dependencies = {stage.stage_id: set(stage.dependencies) for stage in stages}
        stage_map = {stage.stage_id: stage for stage in stages}
        
        # Build execution graph
        graph = []
        remaining = set(stage_map.keys())
        
        while remaining:
            # Find stages with no dependencies
            batch = {
                stage_id for stage_id in remaining
                if not dependencies[stage_id]
            }
            
            if not batch:
                raise ValueError("Circular dependency detected")
            
            # Add stage batch
            graph.append([stage_map[stage_id] for stage_id in batch])
            
            # Update remaining stages
            remaining -= batch
            
            # Update dependencies
            for deps in dependencies.values():
                deps -= batch
        
        return graph
    
    async def _save_workflow(
        self,
        workflow: WorkflowResult
    ) -> None:
        """Save workflow results.
        
        Args:
            workflow: Workflow results
        """
        try:
            workflow_file = self.workflow_dir / f"{workflow.workflow_id}.json"
            
            # Create backup if exists
            if workflow_file.exists():
                backup_file = workflow_file.with_suffix(".bak")
                workflow_file.rename(backup_file)
            
            # Save workflow
            with open(workflow_file, "w") as f:
                json.dump(
                    workflow.dict(),
                    f,
                    indent=2,
                    default=str
                )
                
        except Exception as e:
            self.error_handler.handle_error(
                "WORKFLOW_ERROR",
                f"Failed to save workflow: {e}",
                details={"workflow_id": workflow.workflow_id}
            )
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Cancel active stages
            for task in self.active_stages.values():
                task.cancel()
            
            # Cleanup components
            self.pipeline.cleanup()
            
            # Cleanup monitor
            if self.monitor:
                self.monitor.cleanup()
                
        except Exception as e:
            self.error_handler.handle_error(
                "CLEANUP_ERROR",
                f"Failed to clean up resources: {e}"
            ) 