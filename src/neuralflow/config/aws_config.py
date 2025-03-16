"""AWS configuration settings for NeuralFlow."""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NetworkConfig(BaseModel):
    """Network configuration settings."""
    
    vpc_id: str = Field(default=os.getenv("SAGEMAKER_VPC_ID", "vpc-0a87459cb380bbdee"))
    subnets: List[str] = Field(default_factory=lambda: [
        "subnet-0b77f2ba7ea3add36",
        "subnet-0238e00b9ed8017d7",
        "subnet-004e61d766db435e9"
    ])
    security_groups: List[str] = Field(
        default_factory=lambda: os.getenv("AWS_VPC_SECURITY_GROUPS", "").split(",") if os.getenv("AWS_VPC_SECURITY_GROUPS") else []
    )
    network_mode: str = Field(default="Public")

class IAMConfig(BaseModel):
    """IAM configuration settings."""
    
    default_execution_role: str = Field(
        default=os.getenv(
            "SAGEMAKER_EXECUTION_ROLE",
            "arn:aws:iam::927450005963:role/service-role/AmazonSageMaker-ExecutionRole-20250316T183769"
        )
    )
    space_execution_role: str = Field(
        default=os.getenv(
            "SAGEMAKER_SPACE_EXECUTION_ROLE",
            "arn:aws:iam::927450005963:role/service-role/AmazonSageMaker-ExecutionRole-20250316T183769"
        )
    )

class SageMakerDomainConfig(BaseModel):
    """SageMaker Domain configuration settings."""
    
    domain_id: str = Field(default=os.getenv("SAGEMAKER_DOMAIN_ID", "d-dmrlqv7dxicz"))
    domain_name: str = Field(default=os.getenv("SAGEMAKER_DOMAIN_NAME", "QuickSetupDomain-20250316T183768"))
    status: str = Field(default="Ready")
    creation_time: str = Field(default="2025-03-16T18:38:01")
    last_modified_time: str = Field(default="2025-03-16T18:43:44")
    
    # Network and IAM configurations
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    iam: IAMConfig = Field(default_factory=IAMConfig)
    
    # Security settings
    auth_mode: str = Field(default="IAM")
    encryption_key_id: Optional[str] = Field(default=os.getenv("AWS_KMS_KEY_ID"))

class AWSConfig(BaseModel):
    """AWS configuration settings."""
    
    # AWS credentials and region
    aws_access_key_id: Optional[str] = Field(default=os.getenv("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key: Optional[str] = Field(default=os.getenv("AWS_SECRET_ACCESS_KEY"))
    aws_region: str = Field(default=os.getenv("AWS_REGION", "us-west-2"))
    
    # SageMaker Domain settings
    domain: SageMakerDomainConfig = Field(default_factory=SageMakerDomainConfig)
    
    # SageMaker settings
    sagemaker_role: str = Field(
        default=os.getenv(
            "SAGEMAKER_ROLE",
            "arn:aws:iam::927450005963:role/service-role/AmazonSageMaker-ExecutionRole-20250316T183769"
        )
    )
    instance_type: str = Field(default=os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.p3.2xlarge"))
    instance_count: int = Field(default=int(os.getenv("SAGEMAKER_INSTANCE_COUNT", "1")))
    
    # S3 bucket settings
    s3_bucket: str = Field(default=os.getenv("AWS_S3_BUCKET"))
    s3_prefix: str = Field(default="neuralflow")
    
    # Framework settings
    framework_version: str = Field(default="2.10.0")
    py_version: str = Field(default="py39")
    
    # Security settings
    kms_key_id: Optional[str] = Field(default=os.getenv("AWS_KMS_KEY_ID"))
    enable_network_isolation: bool = Field(default=True)
    enable_inter_container_traffic_encryption: bool = Field(default=True)
    
    # Monitoring settings
    enable_cloudwatch: bool = Field(default=True)
    cloudwatch_log_group: Optional[str] = Field(default=os.getenv("AWS_CLOUDWATCH_LOG_GROUP", "/aws/sagemaker/training-jobs"))
    enable_container_insights: bool = Field(default=True)
    
    # Tags
    tags: Dict[str, str] = Field(default_factory=lambda: {
        "Environment": os.getenv("AWS_ENVIRONMENT", "development"),
        "Project": "neuralflow",
        "ManagedBy": "sagemaker-sdk",
        "SecurityCompliance": "enabled"
    })
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False 