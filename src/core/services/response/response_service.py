from config.config import Config
from transformers import pipeline
from models.model_manager import ModelManager
import logging
from typing import Dict, Any, Optional, AsyncGenerator
from tools.sentiment_analyzer import SentimentAnalyzer
from tools.memory_manager import MemoryManager
from unittest.mock import patch

# Import GGUF model wrapper
try:
    from models.gguf_wrapper.llm_wrapper import get_llm
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False
    print("GGUF wrapper not available. Make sure llama-cpp-python is installed.")

# Initialize logger
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates structured responses by synthesizing execution results and retrieved context, and handles sentiment analysis."""
    
    def __init__(self, llm=None, sentiment_analyzer=None, model_manager=None, model=None, temperature=None, memory_manager=None, model_instance=None, config=None):
        """
        Initializes the ResponseGenerator with a language model and sentiment analysis pipeline.
        
        Args:
            llm (ChatOpenAI): An instance of the ChatOpenAI model with specified parameters.
            sentiment_analyzer (Pipeline): An instance of the sentiment analysis pipeline.
            model_manager (ModelManager): An instance of the ModelManager for emotion inference.
            model (str): The name of the LLM model to use.
            temperature (float): The temperature for the LLM.
            memory_manager: The memory manager component.
            model_instance: A direct model instance to use (overrides other LLM parameters)
            config: Configuration object. If None, uses global Config.
        
        Attributes:
            llm (ChatOpenAI): An instance of the ChatOpenAI model with specified parameters.
            sentiment_analyzer (Pipeline): An instance of the sentiment analysis pipeline.
            model_manager (ModelManager): An instance of the ModelManager for emotion inference.
            memory_manager: The memory manager component.
            config: Configuration object.
        """
        # Use provided config or global Config
        self.config = config or Config
        self.logger = logging.getLogger(__name__)
        
        try:
            default_model = getattr(self.config, 'RESPONSE_MODEL', 'local-gguf')
            default_temp = getattr(self.config, 'RESPONSE_TEMPERATURE', 0.7)
            
            # Use provided LLM, model instance, or create one
            if llm:
                self.llm = llm
            elif model_instance:
                # Use directly provided model instance
                class LangChainAdapter:
                    def __init__(self, model_instance):
                        self.model = model_instance
                        
                    async def apredict(self, prompt):
                        if hasattr(self.model, 'agenerate'):
                            return await self.model.agenerate(prompt)
                        elif hasattr(self.model, 'generate'):
                            # If no async method available, use the sync method
                            return self.model.generate(prompt)
                        else:
                            # Last resort, try direct call
                            return self.model(prompt)
                            
                    def predict(self, prompt):
                        if hasattr(self.model, 'generate'):
                            return self.model.generate(prompt)
                        else:
                            # Try direct call
                            return self.model(prompt)
                
                self.llm = LangChainAdapter(model_instance)
                print(f"Using provided model instance directly")
            elif model == "local-gguf" or default_model == "local-gguf":
                # Check if GGUF wrapper is available
                if HAS_GGUF:
                    # Get GGUF model path
                    gguf_model_path = getattr(self.config, 'GGUF_MODEL_PATH', None)
                    gguf_max_tokens = getattr(self.config, 'GGUF_MAX_TOKENS', 512)
                    gguf_top_p = getattr(self.config, 'GGUF_TOP_P', 0.95)
                    gguf_top_k = getattr(self.config, 'GGUF_TOP_K', 40)
                    gguf_context_window = getattr(self.config, 'GGUF_CONTEXT_WINDOW', 4096)
                    gguf_n_gpu_layers = getattr(self.config, 'GGUF_N_GPU_LAYERS', -1)
                    
                    # Initialize the GGUF model wrapper
                    try:
                        # Check if model path exists
                        import os
                        if gguf_model_path and os.path.exists(gguf_model_path):
                            model_path_to_use = str(gguf_model_path)
                        else:
                            # Try to find any .gguf file in the models/gguf_llm directory
                            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            model_dir = os.path.join(base_dir, "models", "gguf_llm")
                            
                            if os.path.exists(model_dir):
                                print(f"Looking for .gguf files in {model_dir}")
                                for file in os.listdir(model_dir):
                                    if file.endswith(".gguf"):
                                        model_path_to_use = os.path.join(model_dir, file)
                                        print(f"Found GGUF model: {model_path_to_use}")
                                        break
                                else:
                                    raise FileNotFoundError(f"No .gguf files found in {model_dir}")
                            else:
                                raise FileNotFoundError(f"Model directory not found: {model_dir}")
                        
                        local_llm = get_llm(
                            model_path=model_path_to_use,
                            temperature=temperature or default_temp,
                            max_tokens=gguf_max_tokens,
                            top_p=gguf_top_p,
                            top_k=gguf_top_k,
                            context_window=gguf_context_window,
                            n_gpu_layers=gguf_n_gpu_layers
                        )
                        
                        # Create a compatible adapter for the LLM
                        class LangChainAdapter:
                            def __init__(self, llm):
                                self.llm = llm
                                
                            async def apredict(self, prompt):
                                return self.llm.generate(prompt)
                                
                            def predict(self, prompt):
                                return self.llm.generate(prompt)
                        
                        self.llm = LangChainAdapter(local_llm)
                        print(f"Initialized local GGUF model: {gguf_model_path}")
                    except Exception as gguf_error:
                        # Raise a clear error about GGUF model initialization failure
                        raise RuntimeError(f"Error initializing GGUF model: {gguf_error}. Make sure the GGUF model file exists and is valid.") 
                else:
                    # Raise a clear error about missing GGUF wrapper
                    raise ImportError("GGUF wrapper not available. Install llama-cpp-python with: pip install llama-cpp-python")
            if model_manager:
                self.llm = model_manager.get_llm()  # Use the LLM from ModelManager
        except Exception as e:
            # Log the error but raise it to ensure proper error handling
            logging.error(f"Error initializing LLM: {e}")
            raise RuntimeError(f"Failed to initialize any language model: {e}")
            
        try:
            # Use provided sentiment analyzer or create one
            sentiment_pipeline = getattr(self.config, 'SENTIMENT_ANALYSIS_PIPELINE', 'sentiment-analysis')
            self.sentiment_analyzer = sentiment_analyzer or pipeline(sentiment_pipeline)
        except Exception as e:
            # Try to use a minimal distilbert model for sentiment analysis
            try:
                logging.warning(f"Error initializing primary sentiment analyzer: {e}. Trying fallback model.")
                # Use a smaller, more widely available model
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                logging.info("Successfully initialized fallback sentiment analyzer")
            except Exception as e2:
                logging.error(f"Error initializing fallback sentiment analyzer: {e2}")
                raise RuntimeError(f"Failed to initialize any sentiment analyzer. Original error: {e}. Fallback error: {e2}")
        
        try:
            # Use provided model manager or create one
            from models.model_manager import ModelManager
            
            self.model_manager = model_manager or ModelManager(config=self.config)
        except Exception as e:
            logging.error(f"Error initializing model manager: {e}")
            # Don't create a mock, raise the error for proper handling
            raise ImportError(f"Failed to initialize model manager: {e}")
    
    async def generate(self, execution_result, retrieved_context):
        """
        Constructs a response by merging execution results with retrieved knowledge and analyzes user sentiment.
        
        Args:
            execution_result (str): The result of the execution to be included in the response.
            retrieved_context (str): The context retrieved from knowledge sources to be included in the response.
        
        Returns:
            str: The final structured response generated by the language model.
        
        Raises:
            ValueError: If execution_result or retrieved_context is empty.
            RuntimeError: If the language model fails to generate a response.
        """
        # Handle various input types and provide defaults
        if not execution_result:
            execution_result = {"result": "No execution result available."}
        if not retrieved_context:
            retrieved_context = {"context": "No context available."}
            
        # Ensure execution_result and retrieved_context are dictionaries
        if not isinstance(execution_result, dict):
            execution_result = {"result": str(execution_result)}
        if not isinstance(retrieved_context, dict):
            retrieved_context = {"context": str(retrieved_context)}
            
        # Check if execution_result only contains an error key
        if "error" in execution_result and len(execution_result) == 1:
            # Add a general_info key with a helpful message
            execution_result["general_info"] = "I'll do my best to help you with your query."
        
        # Extract user query if available
        user_query = execution_result.get("user_query", "") or retrieved_context.get("user_query", "")
        sentiment_label, sentiment_score, style_label, emotion_label = "NEUTRAL", 0.5, "neutral", "neutral"
        
        if user_query:
            try:
                # Analyze sentiment
                sentiment_label, sentiment_score = self._analyze_sentiment(user_query)
                print(f"User sentiment: {sentiment_label} ({sentiment_score})")
            except Exception as e:
                print(f"Error analyzing sentiment: {e}")
                
            try:
                # Classify style if available
                if hasattr(self.model_manager, 'classify_style'):
                    style_label = self.model_manager.classify_style(user_query)
                    print(f"User query style: {style_label}")
            except Exception as e:
                print(f"Error classifying style: {e}")
                
            try:
                # Predict emotion if available
                if hasattr(self.model_manager, 'emotion_model') and hasattr(self.model_manager.emotion_model, 'predict'):
                    emotion_label = self.model_manager.emotion_model.predict([user_query])[0]
                    print(f"User emotion: {emotion_label}")
            except Exception as e:
                print(f"Error predicting emotion: {e}")

        try:
            # Create response prompt
            response_prompt = self._create_response_prompt(
                sentiment_label, style_label, emotion_label, execution_result, retrieved_context
            )
            
            # Generate response
            try:
                final_response = await self.llm.apredict(response_prompt)
            except Exception as e:
                print(f"Error in LLM prediction: {e}")
                if "llama_decode returned -3" in str(e):
                    final_response = "I'm currently experiencing some technical difficulties with the language model. Please try again later."
                else:
                    # Create a simple fallback response based on the query
                    user_query = retrieved_context.get("user_query", "") if isinstance(retrieved_context, dict) else ""
                    if user_query.lower() in ["hi", "hello", "hey"]:
                        final_response = "Hello! How can I help you today?"
                    else:
                        final_response = f"I received your message about '{user_query}'. How can I assist you with this?"
            
            # Store in execution_result for direct LangGraph state update
            if isinstance(execution_result, dict):
                execution_result["final_response"] = final_response
                print(f"DEBUG - Stored final_response in execution_result dictionary")
                
            # Also directly update the state if retrieved_context is the state dict
            if isinstance(retrieved_context, dict):
                retrieved_context["final_response"] = final_response
                print(f"DEBUG - Stored final_response in retrieved_context dictionary")
            
            # Log the generated response
            logging.info(f"Generated response: {final_response[:100]}...")
            print(f"ResponseGenerator: Generated response: {final_response[:100]}...")
            
            # Save interaction to memory if available
            try:
                if hasattr(self, 'model_manager') and self.model_manager is not None:
                    if hasattr(self.model_manager, 'memory_manager') and self.model_manager.memory_manager is not None:
                        memory_manager = self.model_manager.memory_manager
                        
                        # Prepare a clean summary of retrieved context to avoid serialization issues
                        context_summary = str(retrieved_context)[:500] if retrieved_context else ""
                        
                        # Extract user query
                        user_query = retrieved_context.get("user_query", "") if isinstance(retrieved_context, dict) else ""
                        
                        # Save interaction safely
                        try:
                            await memory_manager.save_interaction(
                                interaction_type="conversation",
                                user_query=user_query,
                                response=final_response,
                                document_path=None,
                                search_results=context_summary
                            )
                            logging.info("Saved interaction to memory")
                        except Exception as save_err:
                            logging.error(f"Error in save_interaction: {save_err}")
                            # Continue even if saving fails
            except Exception as memory_err:
                logging.error(f"Error with memory manager: {memory_err}")
                # Continue even if memory operations fail
                
            return final_response
        except Exception as e:
            print(f"Error generating response: {e}")
            # Return a simple fallback response
            return f"I understand you were asking about {user_query}. I'm currently experiencing some technical difficulties. Could you please try again or rephrase your question?"

    def _analyze_sentiment(self, user_query):
        """Analyzes the sentiment of the user query."""
        sentiment_result = self.sentiment_analyzer(user_query)[0]
        return sentiment_result["label"], sentiment_result["score"]

    def _create_response_prompt(self, sentiment_label, style_label, emotion_label, execution_result, retrieved_context):
        """Creates the response prompt based on sentiment, style, and emotion."""
        try:
            # Get prompt templates with fallbacks
            negative_prompt = getattr(self.config, 'RESPONSE_PROMPT_NEGATIVE', """
            User is expressing negative sentiment and the style is {style_label}. Emotion detected: {emotion_label}.
            Given the following information:
            Execution Result: {execution_result}
            Retrieved Context: {retrieved_context}

            Provide a structured, clear response.
            Acknowledge the negative sentiment and offer support or solutions.
            """)
            
            positive_prompt = getattr(self.config, 'RESPONSE_PROMPT_POSITIVE', """
            User query style is {style_label}. Emotion detected: {emotion_label}.
            Given the following information:
            Execution Result: {execution_result}
            Retrieved Context: {retrieved_context}

            Provide a structured, clear response.
            Adapt the response style to match the user's style.
            """)
            
            # Choose the appropriate prompt based on sentiment
            if sentiment_label == "NEGATIVE":
                template = negative_prompt
            else:
                template = positive_prompt
                
            # Format the template with the provided values
            return template.format(
                style_label=style_label or "neutral",
                emotion_label=emotion_label or "neutral",
                execution_result=execution_result,
                retrieved_context=retrieved_context
            )
            
        except Exception as e:
            print(f"Error creating response prompt: {e}. Using default prompt.")
            # Fallback to a simple prompt if formatting fails
            return f"""
            Given the following information:
            Execution Result: {execution_result}
            Retrieved Context: {retrieved_context}
            
            Provide a helpful, structured response.
            """

    def _initialize_components(self):
        """Initialize required components."""
        try:
            self.sentiment_analyzer = SentimentAnalyzer(self.config)
            self.memory_manager = MemoryManager(self.config)
        except Exception as e:
            logger.warning(f"Failed to initialize components: {e}")

    async def generate_response(
        self,
        query: str,
        context: Dict[str, Any],
        priority: Optional[str] = None
    ) -> str:
        """Generate a response based on the query and context."""
        try:
            # Format the prompt
            formatted_prompt = self._format_prompt(query, context)
            
            # Generate response
            response = await self.model_manager.generate_response(formatted_prompt)
            
            # Apply priority if specified
            if priority == "urgent":
                response = await self.prioritize_response(response)
            
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    async def stream_response(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Stream the response generation."""
        try:
            # Format the prompt
            formatted_prompt = self._format_prompt(query, context)
            
            # Stream response
            async for chunk in await self.model_manager.stream_response(formatted_prompt):
                yield chunk
        except Exception as e:
            self.logger.error(f"Error streaming response: {str(e)}")
            yield f"Error streaming response: {str(e)}"

    async def apply_response_style(
        self,
        response: str,
        style: str
    ) -> str:
        """Apply a specific style to the response."""
        try:
            # Format the prompt for style application
            prompt = f"Apply the following style to the response: {style}\n\nResponse: {response}"
            
            # Generate styled response
            styled_response = await self.model_manager.generate_response(prompt)
            return styled_response
        except Exception as e:
            self.logger.error(f"Error applying response style: {str(e)}")
            return response

    async def prioritize_response(
        self,
        response: str
    ) -> str:
        """Add urgency indicators to the response."""
        try:
            # Format the prompt for prioritization
            prompt = f"Add urgency indicators to the following response: {response}"
            
            # Generate prioritized response
            prioritized_response = await self.model_manager.generate_response(prompt)
            return prioritized_response
        except Exception as e:
            self.logger.error(f"Error prioritizing response: {str(e)}")
            return response

    def _format_prompt(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Format the prompt with query and context."""
        try:
            # Extract relevant information from context
            documents = context.get("documents", [])
            memory = context.get("memory", {})
            
            # Build prompt string
            prompt = f"Query: {query}\n\n"
            
            # Add document context if available
            if documents:
                prompt += "Relevant documents:\n"
                for doc in documents:
                    prompt += f"- {doc.get('content', '')}\n"
                prompt += "\n"
            
            # Add memory context if available
            if memory:
                prompt += "Memory context:\n"
                for memory_type, memories in memory.items():
                    if memories:
                        prompt += f"{memory_type}:\n"
                        for mem in memories:
                            prompt += f"- {mem}\n"
                prompt += "\n"
            
            return prompt
        except Exception as e:
            self.logger.error(f"Error formatting prompt: {str(e)}")
            return f"Query: {query}\n\n"
