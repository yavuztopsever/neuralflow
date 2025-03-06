"""
Chat interface component for the LangGraph project.
"""

import gradio as gr
from typing import List, Dict, Any
import json
from config.config import Config
from core.models.management.model_manager import ModelManager

class ChatInterface:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.model_manager = ModelManager(Config)
        self.current_model = "local-gguf"  # Default model
        
    def create_chat_interface(self):
        """Create the chat interface using Gradio."""
        
        def chat(message: str, history: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
            # Update model if changed
            if model != self.current_model:
                self.current_model = model
                self.model_manager.initialize_llm()
            
            # Get response from the selected model
            response = self.model_manager.generate_response(message)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history
        
        def clear_chat():
            self.history = []
            return []
        
        with gr.Blocks(theme=gr.themes.Soft()) as chat_interface:
            gr.Markdown("# LangGraph Chat Interface")
            
            # Add model selection dropdown
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=["local-gguf", "openai", "anthropic", "huggingface"],
                    value="local-gguf",
                    label="Select LLM Provider",
                    info="Choose the language model provider"
                )
            
            chatbot = gr.Chatbot(
                value=self.history,
                height=600,
                show_label=False,
                container=True
            )
            
            with gr.Row():
                with gr.Column(scale=8):
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        show_label=False,
                        container=False
                    )
                with gr.Column(scale=1):
                    send = gr.Button("Send")
                with gr.Column(scale=1):
                    clear = gr.Button("Clear")
            
            send.click(
                chat,
                inputs=[msg, chatbot, model_dropdown],
                outputs=[chatbot]
            ).then(
                lambda: "",
                None,
                msg
            )
            
            clear.click(
                clear_chat,
                None,
                chatbot
            )
            
            msg.submit(
                chat,
                inputs=[msg, chatbot, model_dropdown],
                outputs=[chatbot]
            ).then(
                lambda: "",
                None,
                msg
            )
        
        return chat_interface
    
    def launch(self, share: bool = False):
        """Launch the chat interface."""
        interface = self.create_chat_interface()
        interface.launch(share=share) 