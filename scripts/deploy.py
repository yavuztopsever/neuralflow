#!/usr/bin/env python3
"""
Deployment script for NeuralFlow.
"""

import argparse
import sys
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.neuralflow.tools.monitoring.core.unified_monitor import UnifiedMonitor, MonitoringConfig
from src.neuralflow.tools.state.core.unified_state import UnifiedState, StateConfig

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        if config_path.endswith('.json'):
            return json.load(f)
        return yaml.safe_load(f)

async def deploy(
    environment: str,
    config: Dict[str, Any]
) -> None:
    """Deploy the application.
    
    Args:
        environment: Deployment environment
        config: Deployment configuration
    """
    # Initialize components
    monitor = UnifiedMonitor(MonitoringConfig(
        monitor_id=f"deploy_{environment}",
        monitor_type="deployment"
    ))
    state = UnifiedState(StateConfig(
        persistence_enabled=True
    ))
    
    try:
        # Create deployment directory
        deploy_dir = Path("deploy") / environment
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy necessary files
        await copy_deployment_files(deploy_dir)
        
        # Configure environment
        await configure_environment(deploy_dir, environment, config)
        
        # Record deployment event
        await monitor.record_event({
            'type': 'deployment',
            'environment': environment,
            'status': 'success',
            'config': config
        })
        
        # Save deployment state
        await state.set_state(
            f"deployment_{environment}",
            {
                'environment': environment,
                'config': config,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
        )
        
        print(f"Deployment to {environment} complete!")
        
    except Exception as e:
        # Record failure
        await monitor.record_event({
            'type': 'deployment',
            'environment': environment,
            'status': 'failed',
            'error': str(e)
        })
        raise
    
    finally:
        # Cleanup
        monitor.cleanup()
        state.cleanup()

async def copy_deployment_files(deploy_dir: Path) -> None:
    """Copy files needed for deployment.
    
    Args:
        deploy_dir: Deployment directory
    """
    # Copy source code
    src_dir = Path("src/neuralflow")
    dest_dir = deploy_dir / "src/neuralflow"
    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
    
    # Copy configuration
    config_dir = Path("config")
    dest_config = deploy_dir / "config"
    shutil.copytree(config_dir, dest_config, dirs_exist_ok=True)
    
    # Copy requirements
    requirements = Path("requirements")
    dest_requirements = deploy_dir / "requirements"
    shutil.copytree(requirements, dest_requirements, dirs_exist_ok=True)
    
    # Copy scripts
    scripts = Path("scripts")
    dest_scripts = deploy_dir / "scripts"
    shutil.copytree(scripts, dest_scripts, dirs_exist_ok=True)

async def configure_environment(
    deploy_dir: Path,
    environment: str,
    config: Dict[str, Any]
) -> None:
    """Configure the deployment environment.
    
    Args:
        deploy_dir: Deployment directory
        environment: Deployment environment
        config: Deployment configuration
    """
    # Create environment file
    env_file = deploy_dir / ".env"
    with open(env_file, "w") as f:
        for key, value in config.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    f.write(f"{key.upper()}_{k.upper()}={v}\n")
            else:
                f.write(f"{key.upper()}={value}\n")
    
    # Create necessary directories
    directories = [
        "logs/app",
        "logs/access",
        "logs/error",
        "storage/models",
        "storage/vector_store",
        "storage/cache",
        "storage/state"
    ]
    
    for directory in directories:
        (deploy_dir / directory).mkdir(parents=True, exist_ok=True)

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Deploy NeuralFlow")
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        required=True,
        help="Deployment environment"
    )
    parser.add_argument(
        "--config",
        help="Path to deployment configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    try:
        await deploy(args.environment, config)
    except Exception as e:
        print(f"Error deploying: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
