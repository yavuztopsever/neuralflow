import psutil
import asyncio
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class MemoryThreshold:
    warning: float  # Warning threshold in MB
    critical: float  # Critical threshold in MB
    max_chunk_size: int  # Maximum size of data chunks in MB

class MemoryManager:
    def __init__(self, thresholds: Optional[MemoryThreshold] = None):
        self.thresholds = thresholds or MemoryThreshold(
            warning=1024,  # 1GB warning
            critical=2048,  # 2GB critical
            max_chunk_size=100  # 100MB chunks
        )
        self.process = psutil.Process()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        self._memory_usage_history = []
        self._max_history_size = 1000

    async def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.process.memory_info().rss / (1024 * 1024)
        )

    async def check_memory_status(self) -> Dict[str, Any]:
        """Check current memory status and return warning level."""
        current_usage = await self.get_memory_usage()
        self._memory_usage_history.append(current_usage)
        if len(self._memory_usage_history) > self._max_history_size:
            self._memory_usage_history.pop(0)

        status = {
            "current_usage": current_usage,
            "warning_threshold": self.thresholds.warning,
            "critical_threshold": self.thresholds.critical,
            "status": "normal"
        }

        if current_usage > self.thresholds.critical:
            status["status"] = "critical"
            self.logger.warning(f"Critical memory usage: {current_usage:.2f}MB")
        elif current_usage > self.thresholds.warning:
            status["status"] = "warning"
            self.logger.warning(f"High memory usage: {current_usage:.2f}MB")

        return status

    async def chunk_data(self, data: Any, chunk_size: Optional[int] = None) -> list:
        """Split data into smaller chunks based on memory constraints."""
        chunk_size = chunk_size or self.thresholds.max_chunk_size
        if isinstance(data, (str, bytes)):
            # For strings and bytes, split by size
            return [data[i:i + chunk_size * 1024 * 1024] for i in range(0, len(data), chunk_size * 1024 * 1024)]
        elif isinstance(data, list):
            # For lists, split into smaller sublists
            return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        elif isinstance(data, dict):
            # For dictionaries, split into smaller dictionaries
            items = list(data.items())
            return [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
        else:
            # For other types, return as single chunk
            return [data]

    async def process_in_chunks(self, data: Any, process_func: callable, chunk_size: Optional[int] = None) -> Any:
        """Process data in chunks to manage memory usage."""
        chunks = await self.chunk_data(data, chunk_size)
        results = []

        for chunk in chunks:
            # Check memory before processing each chunk
            memory_status = await self.check_memory_status()
            if memory_status["status"] == "critical":
                # Wait for memory to be freed
                await asyncio.sleep(1)
                memory_status = await self.check_memory_status()
                if memory_status["status"] == "critical":
                    raise MemoryError("Critical memory usage detected")

            # Process chunk
            result = await process_func(chunk)
            results.append(result)

            # Allow other tasks to run between chunks
            await asyncio.sleep(0)

        # Combine results based on data type
        if isinstance(data, (str, bytes)):
            return "".join(results)
        elif isinstance(data, list):
            return [item for sublist in results for item in sublist]
        elif isinstance(data, dict):
            return {k: v for d in results for k, v in d.items()}
        else:
            return results[0]

    async def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        self._memory_usage_history.clear()

    def get_memory_history(self) -> list:
        """Get memory usage history."""
        return self._memory_usage_history.copy() 