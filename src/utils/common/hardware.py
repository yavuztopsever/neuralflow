"""
Hardware detection and optimization utilities for the LangGraph application.
This module provides functionality for detecting the operating system and available hardware,
and optimizing the application accordingly.
"""

import os
import platform
import logging
import torch
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class HardwareManager:
    """Manages hardware detection and optimization."""
    
    def __init__(self):
        """Initialize the hardware manager."""
        self.system = platform.system()
        self.device = self._detect_device()
        self.device_info = self._get_device_info()
        logger.info(f"Initialized hardware manager with device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detect the appropriate device for computation.
        
        Returns:
            str: The detected device ('mps', 'cuda', or 'cpu')
        """
        try:
            if self.system == 'Darwin' and torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except Exception as e:
            logger.warning(f"Error detecting device: {e}. Falling back to CPU.")
            return 'cpu'
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get information about the detected device.
        
        Returns:
            Dict[str, Any]: Device information including name, memory, etc.
        """
        info = {
            'device': self.device,
            'system': self.system,
            'memory': None,
            'cores': None
        }
        
        try:
            if self.device == 'cuda':
                info['memory'] = torch.cuda.get_device_properties(0).total_memory
                info['cores'] = torch.cuda.get_device_properties(0).multi_processor_count
            elif self.device == 'mps':
                # MPS doesn't provide detailed device info, but we can get some system info
                import psutil
                info['memory'] = psutil.virtual_memory().total
                info['cores'] = psutil.cpu_count()
            else:  # cpu
                import psutil
                info['memory'] = psutil.virtual_memory().total
                info['cores'] = psutil.cpu_count()
        except Exception as e:
            logger.warning(f"Error getting device info: {e}")
        
        return info
    
    def optimize_torch_settings(self) -> None:
        """Optimize PyTorch settings based on the detected device."""
        try:
            if self.device == 'cuda':
                # Optimize CUDA settings
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            elif self.device == 'mps':
                # MPS-specific optimizations
                torch.backends.mps.enable_fallback_to_cpu = True
            else:
                # CPU optimizations
                torch.set_num_threads(self.device_info['cores'])
        except Exception as e:
            logger.warning(f"Error optimizing torch settings: {e}")
    
    def get_device_for_model(self, model_name: str) -> Tuple[str, Dict[str, Any]]:
        """Get the appropriate device and settings for a specific model.
        
        Args:
            model_name: Name of the model to optimize for
            
        Returns:
            Tuple[str, Dict[str, Any]]: Device and optimization settings
        """
        settings = {
            'device': self.device,
            'dtype': torch.float32,
            'num_workers': min(4, self.device_info['cores']),
            'pin_memory': self.device == 'cuda'
        }
        
        # Adjust settings based on model requirements
        if 'large' in model_name.lower() and self.device == 'cuda':
            settings['dtype'] = torch.float16  # Use half precision for large models
            settings['num_workers'] = min(2, self.device_info['cores'])
        
        return self.device, settings
    
    def log_hardware_info(self) -> None:
        """Log information about the detected hardware."""
        logger.info(f"System: {self.system}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Memory: {self.device_info['memory'] / (1024**3):.2f} GB")
        logger.info(f"Cores: {self.device_info['cores']}") 