"""
Workflow visualization component for the LangGraph project.
"""

import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import json

class WorkflowVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def create_workflow_interface(self):
        """Create the workflow visualization interface using Gradio."""
        
        def update_graph(graph_data: str) -> str:
            """Update the graph with new data."""
            try:
                data = json.loads(graph_data)
                self.graph.clear()
                for node in data.get('nodes', []):
                    self.graph.add_node(node['id'], **node)
                for edge in data.get('edges', []):
                    self.graph.add_edge(edge['source'], edge['target'])
                
                # Create visualization
                plt.figure(figsize=(10, 6))
                pos = nx.spring_layout(self.graph)
                nx.draw(self.graph, pos, with_labels=True, node_color='lightblue',
                       node_size=2000, font_size=10, font_weight='bold',
                       arrows=True, edge_color='gray')
                
                # Save the plot
                plt.savefig('workflow.png')
                plt.close()
                return 'workflow.png'
            except Exception as e:
                return f"Error: {str(e)}"
        
        with gr.Blocks(theme=gr.themes.Soft()) as workflow_interface:
            gr.Markdown("# LangGraph Workflow Visualization")
            
            with gr.Row():
                with gr.Column(scale=1):
                    graph_data = gr.Textbox(
                        label="Graph Data (JSON)",
                        placeholder='{"nodes": [{"id": "node1", "label": "Node 1"}], "edges": [{"source": "node1", "target": "node2"}]}',
                        lines=10
                    )
                with gr.Column(scale=1):
                    update_btn = gr.Button("Update Graph")
            
            output_image = gr.Image(label="Workflow Visualization")
            
            update_btn.click(
                update_graph,
                inputs=[graph_data],
                outputs=[output_image]
            )
        
        return workflow_interface
    
    def launch(self, share: bool = False):
        """Launch the workflow visualization interface."""
        interface = self.create_workflow_interface()
        interface.launch(share=share) 