"""
Base provider interfaces for database providers.
This module provides base classes for database provider implementations.
"""

import logging
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic, Type
from abc import ABC, abstractmethod
from datetime import datetime
from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

T = TypeVar('T')

class DatabaseConfig:
    """Configuration for database providers."""
    
    def __init__(self,
                 url: str,
                 **kwargs):
        """Initialize the configuration.
        
        Args:
            url: Database URL
            **kwargs: Additional configuration parameters
        """
        self.url = url
        self.extra_params = kwargs

class BaseModel(declarative_base()):
    """Base model for database entities."""
    
    __tablename__ = 'base_models'
    
    id = Column(String, primary_key=True)
    type = Column(String, nullable=False)
    data = Column(JSON, nullable=False)
    created = Column(DateTime, nullable=False)
    modified = Column(DateTime, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary.
        
        Returns:
            Dictionary representation of the model
        """
        return {
            'id': self.id,
            'type': self.type,
            'data': self.data,
            'created': self.created.isoformat(),
            'modified': self.modified.isoformat()
        }

class BaseDatabaseProvider(ABC, Generic[T]):
    """Base class for database providers."""
    
    def __init__(self, provider_id: str,
                 config: DatabaseConfig,
                 model_class: Type[T],
                 **kwargs):
        """Initialize the provider.
        
        Args:
            provider_id: Unique identifier for the provider
            config: Database provider configuration
            model_class: Model class to use
            **kwargs: Additional initialization parameters
        """
        self.id = provider_id
        self.config = config
        self.model_class = model_class
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._kwargs = kwargs
        
        # Initialize SQLAlchemy
        self.engine = create_engine(config.url, **config.extra_params)
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = MetaData()
        
        # Create tables
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create database tables."""
        try:
            BaseModel.metadata.create_all(self.engine)
            logger.info(f"Created database tables for provider {self.id}")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create database tables for provider {self.id}: {e}")
            raise
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up provider resources."""
        pass
    
    def create(self, obj: T) -> bool:
        """Create a new object.
        
        Args:
            obj: Object to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session = self.Session()
            try:
                base_model = BaseModel(
                    id=str(obj.id),
                    type=obj.__class__.__name__,
                    data=obj.to_dict(),
                    created=datetime.now(),
                    modified=datetime.now()
                )
                session.add(base_model)
                session.commit()
                return True
            finally:
                session.close()
        except SQLAlchemyError as e:
            logger.error(f"Failed to create object in provider {self.id}: {e}")
            return False
    
    def get(self, obj_id: str) -> Optional[T]:
        """Get an object by ID.
        
        Args:
            obj_id: Object ID
            
        Returns:
            Object or None if not found
        """
        try:
            session = self.Session()
            try:
                base_model = session.query(BaseModel).filter_by(id=obj_id).first()
                if base_model is None:
                    return None
                
                return self.model_class.from_dict(base_model.data)
            finally:
                session.close()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get object {obj_id} from provider {self.id}: {e}")
            return None
    
    def update(self, obj: T) -> bool:
        """Update an object.
        
        Args:
            obj: Object to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session = self.Session()
            try:
                base_model = session.query(BaseModel).filter_by(id=str(obj.id)).first()
                if base_model is None:
                    return False
                
                base_model.data = obj.to_dict()
                base_model.modified = datetime.now()
                session.commit()
                return True
            finally:
                session.close()
        except SQLAlchemyError as e:
            logger.error(f"Failed to update object in provider {self.id}: {e}")
            return False
    
    def delete(self, obj_id: str) -> bool:
        """Delete an object.
        
        Args:
            obj_id: Object ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session = self.Session()
            try:
                base_model = session.query(BaseModel).filter_by(id=obj_id).first()
                if base_model is None:
                    return False
                
                session.delete(base_model)
                session.commit()
                return True
            finally:
                session.close()
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete object {obj_id} from provider {self.id}: {e}")
            return False
    
    def list(self, **filters) -> List[T]:
        """List objects matching filters.
        
        Args:
            **filters: Filter criteria
            
        Returns:
            List of matching objects
        """
        try:
            session = self.Session()
            try:
                query = session.query(BaseModel)
                if filters:
                    query = query.filter_by(**filters)
                
                base_models = query.all()
                return [self.model_class.from_dict(m.data) for m in base_models]
            finally:
                session.close()
        except SQLAlchemyError as e:
            logger.error(f"Failed to list objects in provider {self.id}: {e}")
            return []
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information.
        
        Returns:
            Dictionary containing provider information
        """
        return {
            'id': self.id,
            'type': type(self).__name__,
            'created': self.created,
            'modified': self.modified,
            'config': {
                'url': self.config.url,
                'extra_params': self.config.extra_params
            },
            'model_class': self.model_class.__name__
        }
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics.
        
        Returns:
            Dictionary containing provider statistics
        """
        try:
            session = self.Session()
            try:
                total_objects = session.query(BaseModel).count()
                object_types = session.query(
                    BaseModel.type,
                    func.count(BaseModel.id)
                ).group_by(BaseModel.type).all()
                
                return {
                    'total_objects': total_objects,
                    'object_types': dict(object_types)
                }
            finally:
                session.close()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get provider stats: {e}")
            return {} 