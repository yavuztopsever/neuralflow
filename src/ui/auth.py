"""
User management component for the LangGraph project.
"""

import gradio as gr
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path
import hashlib
import jwt
from datetime import datetime, timedelta

class UserManager:
    def __init__(self, storage_dir: str = "storage/users"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key")
        self.current_user: Optional[Dict[str, Any]] = None
        
    def _hash_password(self, password: str) -> str:
        """Hash the password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_token(self, username: str) -> str:
        """Generate a JWT token for the user."""
        payload = {
            "username": username,
            "exp": datetime.utcnow() + timedelta(days=1)
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def _save_user(self, username: str, user_data: Dict[str, Any]):
        """Save user data to storage."""
        user_file = self.storage_dir / f"{username}.json"
        with open(user_file, "w") as f:
            json.dump(user_data, f)
    
    def _load_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Load user data from storage."""
        user_file = self.storage_dir / f"{username}.json"
        if user_file.exists():
            with open(user_file, "r") as f:
                return json.load(f)
        return None
    
    def create_auth_interface(self):
        """Create the authentication interface using Gradio."""
        
        def register(username: str, password: str, email: str) -> str:
            """Register a new user."""
            if self._load_user(username):
                return "Username already exists"
            
            user_data = {
                "username": username,
                "password": self._hash_password(password),
                "email": email,
                "preferences": {},
                "created_at": datetime.utcnow().isoformat()
            }
            
            self._save_user(username, user_data)
            return "Registration successful"
        
        def login(username: str, password: str) -> str:
            """Login a user."""
            user_data = self._load_user(username)
            if not user_data or user_data["password"] != self._hash_password(password):
                return "Invalid username or password"
            
            self.current_user = user_data
            token = self._generate_token(username)
            return f"Login successful! Token: {token}"
        
        def update_preferences(preferences: str) -> str:
            """Update user preferences."""
            if not self.current_user:
                return "Please login first"
            
            try:
                new_preferences = json.loads(preferences)
                self.current_user["preferences"].update(new_preferences)
                self._save_user(self.current_user["username"], self.current_user)
                return "Preferences updated successfully"
            except json.JSONDecodeError:
                return "Invalid JSON format"
        
        with gr.Blocks(theme=gr.themes.Soft()) as auth_interface:
            gr.Markdown("# LangGraph User Management")
            
            with gr.Tab("Register"):
                with gr.Row():
                    with gr.Column():
                        reg_username = gr.Textbox(label="Username")
                        reg_password = gr.Textbox(label="Password", type="password")
                        reg_email = gr.Textbox(label="Email")
                        reg_button = gr.Button("Register")
                reg_output = gr.Textbox(label="Registration Status")
            
            with gr.Tab("Login"):
                with gr.Row():
                    with gr.Column():
                        login_username = gr.Textbox(label="Username")
                        login_password = gr.Textbox(label="Password", type="password")
                        login_button = gr.Button("Login")
                login_output = gr.Textbox(label="Login Status")
            
            with gr.Tab("Preferences"):
                with gr.Row():
                    with gr.Column():
                        preferences = gr.Textbox(
                            label="Preferences (JSON)",
                            placeholder='{"theme": "dark", "language": "en"}',
                            lines=5
                        )
                        pref_button = gr.Button("Update Preferences")
                pref_output = gr.Textbox(label="Preferences Status")
            
            reg_button.click(
                register,
                inputs=[reg_username, reg_password, reg_email],
                outputs=[reg_output]
            )
            
            login_button.click(
                login,
                inputs=[login_username, login_password],
                outputs=[login_output]
            )
            
            pref_button.click(
                update_preferences,
                inputs=[preferences],
                outputs=[pref_output]
            )
        
        return auth_interface
    
    def launch(self, share: bool = False):
        """Launch the authentication interface."""
        interface = self.create_auth_interface()
        interface.launch(share=share) 