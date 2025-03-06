"""
Main UI application for the NeuralFlow project.
"""

import gradio as gr
from typing import Dict, Any
from .chat import ChatInterface
from .workflow import WorkflowVisualizer
from .auth import UserManager

class NeuralFlowUI:
    def __init__(self):
        self.chat_interface = ChatInterface()
        self.workflow_visualizer = WorkflowVisualizer()
        self.user_manager = UserManager()
        
    def create_main_interface(self):
        """Create the main interface that integrates all components."""
        
        with gr.Blocks(theme=gr.themes.Soft()) as main_interface:
            gr.Markdown("# NeuralFlow Application")
            
            with gr.Tabs():
                # Chat Interface Tab
                with gr.Tab("Chat"):
                    self.chat_interface.create_chat_interface()
                
                # Workflow Visualization Tab
                with gr.Tab("Workflow"):
                    self.workflow_visualizer.create_workflow_interface()
                
                # User Management Tab
                with gr.Tab("User Management"):
                    self.user_manager.create_auth_interface()
        
        return main_interface
    
    def launch(self, share: bool = False):
        """Launch the main interface."""
        interface = self.create_main_interface()
        interface.launch(share=share)

def main():
    """Main entry point for the UI application."""
    ui = NeuralFlowUI()
    ui.launch(share=True)  # Enable sharing by default

if __name__ == "__main__":
    main() 