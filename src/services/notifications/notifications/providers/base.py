"""
Base provider interfaces for notification providers.
This module provides base classes for notification provider implementations.
"""

import logging
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')

class NotificationType(Enum):
    """Notification types."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    ALERT = "alert"

@dataclass
class Notification:
    """Notification with metadata."""
    
    id: str
    type: NotificationType
    title: str
    message: str
    recipient: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    read: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary.
        
        Returns:
            Dictionary representation of the notification
        """
        return {
            'id': self.id,
            'type': self.type.value,
            'title': self.title,
            'message': self.message,
            'recipient': self.recipient,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {},
            'read': self.read
        }

class NotificationConfig:
    """Configuration for notification providers."""
    
    def __init__(self,
                 max_notifications: Optional[int] = None,
                 retention_days: Optional[int] = None,
                 **kwargs):
        """Initialize the configuration.
        
        Args:
            max_notifications: Maximum number of notifications to store
            retention_days: Number of days to retain notifications
            **kwargs: Additional configuration parameters
        """
        self.max_notifications = max_notifications
        self.retention_days = retention_days
        self.extra_params = kwargs

class BaseNotificationProvider(ABC):
    """Base class for notification providers."""
    
    def __init__(self, provider_id: str,
                 config: NotificationConfig,
                 **kwargs):
        """Initialize the provider.
        
        Args:
            provider_id: Unique identifier for the provider
            config: Notification provider configuration
            **kwargs: Additional initialization parameters
        """
        self.id = provider_id
        self.config = config
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._kwargs = kwargs
        self._notifications: Dict[str, Notification] = {}
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up provider resources."""
        pass
    
    def send(self,
            notification_type: NotificationType,
            title: str,
            message: str,
            recipient: str,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send a notification.
        
        Args:
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            recipient: Notification recipient
            metadata: Optional notification metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            notification = Notification(
                id=str(uuid.uuid4()),
                type=notification_type,
                title=title,
                message=message,
                recipient=recipient,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            # Store notification
            self._notifications[notification.id] = notification
            
            # Check max notifications
            if (self.config.max_notifications is not None and
                len(self._notifications) > self.config.max_notifications):
                # Remove oldest unread notifications first
                sorted_notifications = sorted(
                    self._notifications.values(),
                    key=lambda x: (x.read, x.timestamp)
                )
                while len(self._notifications) > self.config.max_notifications:
                    oldest = sorted_notifications.pop(0)
                    del self._notifications[oldest.id]
            
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to send notification in provider {self.id}: {e}")
            return False
    
    def get_notifications(self,
                         recipient: Optional[str] = None,
                         notification_type: Optional[NotificationType] = None,
                         read: Optional[bool] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Notification]:
        """Get notifications matching criteria.
        
        Args:
            recipient: Optional recipient to filter by
            notification_type: Optional notification type to filter by
            read: Optional read status to filter by
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            
        Returns:
            List of matching notifications
        """
        try:
            notifications = list(self._notifications.values())
            
            if recipient:
                notifications = [n for n in notifications if n.recipient == recipient]
            
            if notification_type:
                notifications = [n for n in notifications if n.type == notification_type]
            
            if read is not None:
                notifications = [n for n in notifications if n.read == read]
            
            if start_time:
                notifications = [n for n in notifications if n.timestamp >= start_time]
            
            if end_time:
                notifications = [n for n in notifications if n.timestamp <= end_time]
            
            return sorted(notifications, key=lambda x: x.timestamp, reverse=True)
        except Exception as e:
            logger.error(f"Failed to get notifications from provider {self.id}: {e}")
            return []
    
    def mark_as_read(self, notification_id: str) -> bool:
        """Mark a notification as read.
        
        Args:
            notification_id: Notification ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if notification_id not in self._notifications:
                return False
            
            notification = self._notifications[notification_id]
            notification.read = True
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to mark notification {notification_id} as read in provider {self.id}: {e}")
            return False
    
    def delete_notification(self, notification_id: str) -> bool:
        """Delete a notification.
        
        Args:
            notification_id: Notification ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if notification_id not in self._notifications:
                return False
            
            del self._notifications[notification_id]
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to delete notification {notification_id} from provider {self.id}: {e}")
            return False
    
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
                'max_notifications': self.config.max_notifications,
                'retention_days': self.config.retention_days,
                'extra_params': self.config.extra_params
            },
            'stats': {
                'total_notifications': len(self._notifications),
                'unread_notifications': len([n for n in self._notifications.values() if not n.read])
            }
        }
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics.
        
        Returns:
            Dictionary containing provider statistics
        """
        try:
            return {
                'total_notifications': len(self._notifications),
                'notification_types': {
                    nt.value: len([n for n in self._notifications.values() if n.type == nt])
                    for nt in NotificationType
                },
                'recipients': {
                    recipient: len([n for n in self._notifications.values() if n.recipient == recipient])
                    for recipient in set(n.recipient for n in self._notifications.values())
                },
                'read_status': {
                    'read': len([n for n in self._notifications.values() if n.read]),
                    'unread': len([n for n in self._notifications.values() if not n.read])
                }
            }
        except Exception as e:
            logger.error(f"Failed to get provider stats: {e}")
            return {} 