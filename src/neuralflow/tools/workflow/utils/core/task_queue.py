import asyncio
from typing import Dict, Any, Callable, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TaskConfig:
    max_concurrent: int = 5
    rate_limit: int = 100  # requests per minute
    timeout: float = 30.0  # seconds
    retry_count: int = 3
    retry_delay: float = 1.0  # seconds

class TaskQueue:
    def __init__(self, config: Optional[TaskConfig] = None):
        self.config = config or TaskConfig()
        self.logger = logging.getLogger(__name__)
        self._queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._rate_limit_tokens = self.config.rate_limit
        self._last_token_refresh = datetime.now()
        self._processing_tasks = set()
        self._task_stats = {
            "completed": 0,
            "failed": 0,
            "retried": 0,
            "avg_processing_time": 0.0
        }

    async def _refresh_rate_limit(self):
        """Refresh rate limit tokens."""
        now = datetime.now()
        time_passed = (now - self._last_token_refresh).total_seconds()
        if time_passed >= 60:
            self._rate_limit_tokens = self.config.rate_limit
            self._last_token_refresh = now

    async def _wait_for_rate_limit(self):
        """Wait if rate limit is exceeded."""
        await self._refresh_rate_limit()
        while self._rate_limit_tokens <= 0:
            await asyncio.sleep(1)
            await self._refresh_rate_limit()
        self._rate_limit_tokens -= 1

    async def _process_task(self, task_id: str, func: Callable, *args, **kwargs):
        """Process a single task with retries."""
        start_time = datetime.now()
        retries = 0

        while retries < self.config.retry_count:
            try:
                async with self._semaphore:
                    await self._wait_for_rate_limit()
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.timeout
                    )
                    processing_time = (datetime.now() - start_time).total_seconds()
                    self._task_stats["completed"] += 1
                    self._task_stats["avg_processing_time"] = (
                        (self._task_stats["avg_processing_time"] * (self._task_stats["completed"] - 1) +
                         processing_time) / self._task_stats["completed"]
                    )
                    return result
            except asyncio.TimeoutError:
                self.logger.warning(f"Task {task_id} timed out")
                retries += 1
                if retries < self.config.retry_count:
                    await asyncio.sleep(self.config.retry_delay)
                    self._task_stats["retried"] += 1
            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {str(e)}")
                retries += 1
                if retries < self.config.retry_count:
                    await asyncio.sleep(self.config.retry_delay)
                    self._task_stats["retried"] += 1

        self._task_stats["failed"] += 1
        raise Exception(f"Task {task_id} failed after {self.config.retry_count} retries")

    async def add_task(self, task_id: str, func: Callable, *args, **kwargs) -> Any:
        """Add a task to the queue and wait for its completion."""
        task = asyncio.create_task(
            self._process_task(task_id, func, *args, **kwargs)
        )
        self._processing_tasks.add(task)
        task.add_done_callback(self._processing_tasks.discard)
        return await task

    async def add_batch(self, tasks: list[tuple[str, Callable, tuple, dict]]) -> list[Any]:
        """Add multiple tasks and wait for all to complete."""
        return await asyncio.gather(*[
            self.add_task(task_id, func, *args, **kwargs)
            for task_id, func, args, kwargs in tasks
        ])

    def get_stats(self) -> Dict[str, Any]:
        """Get current task processing statistics."""
        return {
            **self._task_stats,
            "queue_size": self._queue.qsize(),
            "active_tasks": len(self._processing_tasks),
            "rate_limit_tokens": self._rate_limit_tokens
        }

    async def cleanup(self):
        """Clean up resources and wait for pending tasks."""
        # Wait for all processing tasks to complete
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        
        # Clear the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break 