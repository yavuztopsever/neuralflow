"""
Unit tests for rate limiting service functionality.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from src.core.services.rate_limit.rate_limit_service import RateLimitService

class TestRateLimitService:
    """Test suite for rate limiting service functionality."""
    
    @pytest.fixture
    def rate_limit_service(self):
        """Create a rate limit service for testing."""
        return RateLimitService(
            max_requests=10,
            time_window=60,  # 1 minute
            burst_size=5
        )
    
    @pytest.mark.asyncio
    async def test_rate_limit_initialization(self, rate_limit_service):
        """Test rate limit service initialization."""
        assert rate_limit_service.max_requests == 10
        assert rate_limit_service.time_window == 60
        assert rate_limit_service.burst_size == 5
        assert rate_limit_service._requests == {}
        assert rate_limit_service._bursts == {}
    
    @pytest.mark.asyncio
    async def test_rate_limit_check(self, rate_limit_service):
        """Test rate limit checking functionality."""
        client_id = "test_client"
        
        # Test within limits
        for _ in range(5):
            result = await rate_limit_service.check_rate_limit(client_id)
            assert result["allowed"] is True
            assert result["remaining"] > 0
            assert result["reset_time"] is not None
        
        # Test burst limit
        for _ in range(5):
            result = await rate_limit_service.check_rate_limit(client_id)
            assert result["allowed"] is True
            assert result["remaining"] > 0
            assert result["reset_time"] is not None
        
        # Test exceeding limits
        result = await rate_limit_service.check_rate_limit(client_id)
        assert result["allowed"] is False
        assert result["remaining"] == 0
        assert result["reset_time"] is not None
    
    @pytest.mark.asyncio
    async def test_rate_limit_reset(self, rate_limit_service):
        """Test rate limit reset functionality."""
        client_id = "test_client"
        
        # Fill up requests
        for _ in range(10):
            await rate_limit_service.check_rate_limit(client_id)
        
        # Reset rate limit
        await rate_limit_service.reset_rate_limit(client_id)
        
        # Check if reset worked
        result = await rate_limit_service.check_rate_limit(client_id)
        assert result["allowed"] is True
        assert result["remaining"] == 10
    
    @pytest.mark.asyncio
    async def test_rate_limit_cleanup(self, rate_limit_service):
        """Test rate limit cleanup functionality."""
        client_id = "test_client"
        
        # Add some requests
        await rate_limit_service.check_rate_limit(client_id)
        
        # Cleanup old requests
        await rate_limit_service.cleanup_old_requests()
        
        # Check if cleanup worked
        assert client_id not in rate_limit_service._requests
    
    @pytest.mark.asyncio
    async def test_rate_limit_metrics(self, rate_limit_service):
        """Test rate limit metrics collection."""
        client_id = "test_client"
        
        # Add some requests
        for _ in range(5):
            await rate_limit_service.check_rate_limit(client_id)
        
        # Get metrics
        metrics = await rate_limit_service.get_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "total_requests" in metrics
        assert "blocked_requests" in metrics
        assert "active_clients" in metrics
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, rate_limit_service):
        """Test rate limit error handling."""
        # Test invalid client ID
        with pytest.raises(ValueError):
            await rate_limit_service.check_rate_limit(None)
        
        # Test invalid max requests
        with pytest.raises(ValueError):
            RateLimitService(max_requests=0, time_window=60)
        
        # Test invalid time window
        with pytest.raises(ValueError):
            RateLimitService(max_requests=10, time_window=0)
        
        # Test invalid burst size
        with pytest.raises(ValueError):
            RateLimitService(max_requests=10, time_window=60, burst_size=0)
    
    @pytest.mark.asyncio
    async def test_rate_limit_concurrent_requests(self, rate_limit_service):
        """Test rate limit handling of concurrent requests."""
        client_id = "test_client"
        
        # Create multiple concurrent requests
        async def make_request():
            return await rate_limit_service.check_rate_limit(client_id)
        
        # Execute concurrent requests
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify results
        assert all(result["allowed"] for result in results)
        assert all(result["remaining"] > 0 for result in results) 