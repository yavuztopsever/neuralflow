"""SageMaker trainer module for remote model training."""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, List
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from botocore.exceptions import ClientError
from ...config.aws_config import AWSConfig

# Configure logging
logger = logging.getLogger(__name__)

class SageMakerTrainer:
    """SageMaker trainer for remote model training."""
    
    def __init__(self, config: Optional[AWSConfig] = None):
        """Initialize SageMaker trainer.
        
        Args:
            config: AWS configuration settings.
        """
        self.config = config or AWSConfig()
        self.session = self._initialize_session()
        self.role = self.config.sagemaker_role
        
    def _initialize_session(self) -> sagemaker.Session:
        """Initialize SageMaker session with Domain configuration.
        
        Returns:
            SageMaker session.
        """
        try:
            # Initialize boto3 client with region
            boto_session = boto3.Session(region_name=self.config.aws_region)
            
            # Create SageMaker client
            sagemaker_client = boto_session.client('sagemaker')
            
            # Verify Domain status and IAM roles
            self._verify_domain_and_permissions(sagemaker_client)
            
            # Initialize session with Domain configuration
            return sagemaker.Session(
                boto_session=boto_session,
                default_bucket=self.config.s3_bucket,
                sagemaker_client=sagemaker_client
            )
        except Exception as e:
            logger.error(f"Failed to initialize SageMaker session: {str(e)}")
            raise
    
    def _verify_domain_and_permissions(self, sagemaker_client) -> None:
        """Verify Domain status and IAM permissions.
        
        Args:
            sagemaker_client: Boto3 SageMaker client.
            
        Raises:
            ValueError: If Domain is not in service or permissions are insufficient.
        """
        try:
            # Check Domain status
            domain_response = sagemaker_client.describe_domain(
                DomainId=self.config.domain.domain_id
            )
            if domain_response['Status'] != 'InService':
                raise ValueError(f"SageMaker Domain status is {domain_response['Status']}")
            
            # Verify IAM role permissions
            iam = boto3.client('iam')
            try:
                iam.get_role(RoleName=self.config.sagemaker_role.split('/')[-1])
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchEntity':
                    raise ValueError(f"IAM role not found: {self.config.sagemaker_role}")
                raise
            
            # Verify VPC configuration
            ec2 = boto3.client('ec2')
            
            # Check VPC exists
            vpc_response = ec2.describe_vpcs(VpcIds=[self.config.domain.network.vpc_id])
            if not vpc_response['Vpcs']:
                raise ValueError(f"VPC not found: {self.config.domain.network.vpc_id}")
            
            # Check subnets exist
            subnet_response = ec2.describe_subnets(
                SubnetIds=self.config.domain.network.subnets
            )
            if len(subnet_response['Subnets']) != len(self.config.domain.network.subnets):
                raise ValueError("One or more subnets not found")
            
            logger.info("Domain and permissions verification successful")
            
        except Exception as e:
            logger.error(f"Domain and permissions verification failed: {str(e)}")
            raise
    
    def _get_vpc_config(self) -> Dict[str, List[str]]:
        """Get VPC configuration for training jobs.
        
        Returns:
            Dict containing VPC configuration.
        """
        return {
            'Subnets': self.config.domain.network.subnets,
            'SecurityGroupIds': self.config.domain.network.security_groups
        }
    
    def _get_security_config(self) -> Dict:
        """Get security configuration for training jobs.
        
        Returns:
            Dict containing security configuration.
        """
        return {
            'EnableNetworkIsolation': self.config.enable_network_isolation,
            'EnableInterContainerTrafficEncryption': self.config.enable_inter_container_traffic_encryption,
            'VpcConfig': self._get_vpc_config(),
            'EnableManagedSpotTraining': False,  # Disable spot instances for security
            'EnableManagedWarmpool': False  # Disable warmpool for security
        }
    
    def upload_data(
        self,
        local_path: Union[str, Path],
        s3_key_prefix: str
    ) -> str:
        """Upload data to S3 bucket with validation.
        
        Args:
            local_path: Local path to data.
            s3_key_prefix: S3 key prefix for uploaded data.
            
        Returns:
            S3 path to uploaded data.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_path}")
            
        try:
            return self.session.upload_data(
                path=str(local_path),
                bucket=self.config.s3_bucket,
                key_prefix=os.path.join(self.config.s3_prefix, s3_key_prefix)
            )
        except Exception as e:
            logger.error(f"Failed to upload data: {str(e)}")
            raise
    
    def create_estimator(
        self,
        entry_point: str,
        hyperparameters: Optional[Dict] = None,
        **kwargs
    ) -> TensorFlow:
        """Create SageMaker TensorFlow estimator with enhanced security.
        
        Args:
            entry_point: Path to training script.
            hyperparameters: Training hyperparameters.
            **kwargs: Additional estimator arguments.
            
        Returns:
            SageMaker TensorFlow estimator.
        """
        try:
            # Get security configuration
            security_config = self._get_security_config()
            
            estimator = TensorFlow(
                entry_point=entry_point,
                role=self.role,
                instance_type=self.config.instance_type,
                instance_count=self.config.instance_count,
                framework_version=self.config.framework_version,
                py_version=self.config.py_version,
                script_mode=True,
                hyperparameters=hyperparameters or {},
                sagemaker_session=self.session,
                enable_cloudwatch_metrics=self.config.enable_cloudwatch,
                enable_container_insights=self.config.enable_container_insights,
                tags=self.config.tags,
                **security_config,
                **kwargs
            )
            
            # Add Domain and security tags
            estimator.tags.update({
                'SageMakerDomain': self.config.domain.domain_name,
                'DomainId': self.config.domain.domain_id,
                'NetworkMode': self.config.domain.network.network_mode,
                'SecurityCompliance': 'enabled'
            })
            
            return estimator
        except Exception as e:
            logger.error(f"Failed to create estimator: {str(e)}")
            raise
    
    def train(
        self,
        estimator: TensorFlow,
        train_data: str,
        validation_data: Optional[str] = None,
        wait: bool = True,
        logs: bool = True
    ) -> None:
        """Start training job with enhanced monitoring and security.
        
        Args:
            estimator: SageMaker estimator.
            train_data: S3 path to training data.
            validation_data: S3 path to validation data.
            wait: Whether to wait for training job completion.
            logs: Whether to show training logs.
        """
        try:
            # Validate data paths and encryption
            self._validate_data_security(train_data)
            if validation_data:
                self._validate_data_security(validation_data)
            
            # Configure inputs
            inputs = {'train': train_data}
            if validation_data:
                inputs['validation'] = validation_data
            
            # Start training job
            job_name = f"neuralflow-{int(time.time())}"
            logger.info(f"Starting training job: {job_name}")
            
            estimator.fit(
                inputs,
                wait=wait,
                logs=logs,
                job_name=job_name
            )
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def _validate_data_security(self, s3_path: str) -> None:
        """Validate S3 path exists and is properly encrypted.
        
        Args:
            s3_path: S3 path to validate.
            
        Raises:
            ValueError: If data validation fails.
        """
        try:
            s3 = boto3.client('s3')
            bucket, key = s3_path.replace('s3://', '').split('/', 1)
            
            # Check if object exists
            try:
                response = s3.head_object(Bucket=bucket, Key=key)
            except ClientError:
                raise ValueError(f"Data not found at: {s3_path}")
            
            # Verify encryption
            if not response.get('ServerSideEncryption'):
                raise ValueError(f"Data at {s3_path} is not encrypted")
            
            # Verify KMS encryption if key is specified
            if self.config.kms_key_id:
                if response.get('SSEKMSKeyId') != self.config.kms_key_id:
                    raise ValueError(f"Data at {s3_path} is not encrypted with the specified KMS key")
            
        except Exception as e:
            logger.error(f"Data security validation failed: {str(e)}")
            raise
    
    def download_model(
        self,
        estimator: TensorFlow,
        local_path: Union[str, Path],
        model_name: Optional[str] = None
    ) -> None:
        """Download trained model artifacts with enhanced security metadata.
        
        Args:
            estimator: SageMaker estimator.
            local_path: Local path to save model artifacts.
            model_name: Optional model name for versioning.
        """
        try:
            model_prefix = os.path.join(
                self.config.s3_prefix,
                'models',
                model_name or estimator.latest_training_job.name
            )
            
            local_path = Path(local_path)
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Download model artifacts
            estimator.model_data = self.session.download_data(
                path=str(local_path),
                bucket=self.config.s3_bucket,
                key_prefix=model_prefix
            )
            
            # Save model metadata with security information
            metadata = {
                'training_job_name': estimator.latest_training_job.name,
                'model_data_path': estimator.model_data,
                'framework_version': self.config.framework_version,
                'python_version': self.config.py_version,
                'domain_id': self.config.domain.domain_id,
                'domain_name': self.config.domain.domain_name,
                'network': {
                    'vpc_id': self.config.domain.network.vpc_id,
                    'subnets': self.config.domain.network.subnets,
                    'network_mode': self.config.domain.network.network_mode
                },
                'security': {
                    'auth_mode': self.config.domain.auth_mode,
                    'network_isolation': self.config.enable_network_isolation,
                    'container_encryption': self.config.enable_inter_container_traffic_encryption,
                    'kms_key_id': self.config.kms_key_id
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with open(local_path / 'model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Model artifacts downloaded to: {local_path}")
            
        except Exception as e:
            logger.error(f"Model download failed: {str(e)}")
            raise 