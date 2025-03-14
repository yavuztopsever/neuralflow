"""
Comprehensive validation rules for different data types.
"""

from typing import Dict, Any, List, Optional, Union, Type
from datetime import datetime
import re
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ValidationRule(BaseModel):
    """Base validation rule."""
    name: str
    description: str
    data_type: str
    severity: str = "error"  # error, warning, info
    parameters: Dict[str, Any] = Field(default_factory=dict)

class TextValidationRules:
    """Text data validation rules."""
    
    @staticmethod
    def length_rule(min_length: int = 1, max_length: int = 1000) -> ValidationRule:
        """Create text length validation rule."""
        return ValidationRule(
            name="text_length",
            description="Validates text length",
            data_type="text",
            parameters={
                "min_length": min_length,
                "max_length": max_length
            }
        )
    
    @staticmethod
    def language_rule(allowed_languages: List[str]) -> ValidationRule:
        """Create language validation rule."""
        return ValidationRule(
            name="language",
            description="Validates text language",
            data_type="text",
            parameters={
                "allowed_languages": allowed_languages
            }
        )
    
    @staticmethod
    def content_quality_rule(
        min_words: int = 3,
        max_duplicate_ratio: float = 0.3
    ) -> ValidationRule:
        """Create content quality validation rule."""
        return ValidationRule(
            name="content_quality",
            description="Validates text content quality",
            data_type="text",
            parameters={
                "min_words": min_words,
                "max_duplicate_ratio": max_duplicate_ratio
            }
        )

class NumericValidationRules:
    """Numeric data validation rules."""
    
    @staticmethod
    def range_rule(
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> ValidationRule:
        """Create numeric range validation rule."""
        return ValidationRule(
            name="numeric_range",
            description="Validates numeric range",
            data_type="numeric",
            parameters={
                "min_value": min_value,
                "max_value": max_value
            }
        )
    
    @staticmethod
    def outlier_rule(
        std_threshold: float = 3.0,
        method: str = "zscore"
    ) -> ValidationRule:
        """Create outlier detection rule."""
        return ValidationRule(
            name="outlier_detection",
            description="Detects numeric outliers",
            data_type="numeric",
            parameters={
                "std_threshold": std_threshold,
                "method": method
            }
        )

class TimeSeriesValidationRules:
    """Time series data validation rules."""
    
    @staticmethod
    def sampling_rate_rule(
        min_rate: float,
        max_rate: float
    ) -> ValidationRule:
        """Create sampling rate validation rule."""
        return ValidationRule(
            name="sampling_rate",
            description="Validates time series sampling rate",
            data_type="time_series",
            parameters={
                "min_rate": min_rate,
                "max_rate": max_rate
            }
        )
    
    @staticmethod
    def completeness_rule(
        max_missing_ratio: float = 0.1
    ) -> ValidationRule:
        """Create data completeness validation rule."""
        return ValidationRule(
            name="completeness",
            description="Validates time series completeness",
            data_type="time_series",
            parameters={
                "max_missing_ratio": max_missing_ratio
            }
        )

class ImageValidationRules:
    """Image data validation rules."""
    
    @staticmethod
    def dimension_rule(
        min_width: int,
        min_height: int,
        max_width: int,
        max_height: int
    ) -> ValidationRule:
        """Create image dimension validation rule."""
        return ValidationRule(
            name="image_dimensions",
            description="Validates image dimensions",
            data_type="image",
            parameters={
                "min_width": min_width,
                "min_height": min_height,
                "max_width": max_width,
                "max_height": max_height
            }
        )
    
    @staticmethod
    def format_rule(
        allowed_formats: List[str]
    ) -> ValidationRule:
        """Create image format validation rule."""
        return ValidationRule(
            name="image_format",
            description="Validates image format",
            data_type="image",
            parameters={
                "allowed_formats": allowed_formats
            }
        )
    
    @staticmethod
    def quality_rule(
        min_resolution: int,
        max_compression: float
    ) -> ValidationRule:
        """Create image quality validation rule."""
        return ValidationRule(
            name="image_quality",
            description="Validates image quality",
            data_type="image",
            parameters={
                "min_resolution": min_resolution,
                "max_compression": max_compression
            }
        )

# Default validation rule sets
DEFAULT_TEXT_RULES = [
    TextValidationRules.length_rule(),
    TextValidationRules.language_rule(["en"]),
    TextValidationRules.content_quality_rule()
]

DEFAULT_NUMERIC_RULES = [
    NumericValidationRules.range_rule(),
    NumericValidationRules.outlier_rule()
]

DEFAULT_TIME_SERIES_RULES = [
    TimeSeriesValidationRules.sampling_rate_rule(0.1, 1000.0),
    TimeSeriesValidationRules.completeness_rule()
]

DEFAULT_IMAGE_RULES = [
    ImageValidationRules.dimension_rule(32, 32, 4096, 4096),
    ImageValidationRules.format_rule(["jpg", "png", "webp"]),
    ImageValidationRules.quality_rule(72, 0.9)
] 