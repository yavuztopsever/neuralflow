from transformers import pipeline
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.model_manager import ModelManager 
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import re
import os
from sklearn.model_selection import train_test_split
from config.config import Config

class ContentRecommendationModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, Config.EMBEDDING_DIM)
        self.fc1 = nn.Linear(Config.EMBEDDING_DIM, Config.HIDDEN_DIM)
        self.fc2 = nn.Linear(Config.HIDDEN_DIM, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class ContentDataset(Dataset):
    def __init__(self, interactions, content_data):
        self.interactions = interactions
        self.content_data = content_data

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        user_query = interaction["user_query"]
        content = interaction["content"]
        label = 1.0  # Assuming positive interaction
        
        # Convert to tensors if not already tensors
        if not isinstance(user_query, torch.Tensor):
            user_query = torch.tensor(user_query) if isinstance(user_query, list) else torch.tensor([0])
        
        if not isinstance(content, torch.Tensor):
            content = torch.tensor(content) if isinstance(content, list) else torch.tensor([0])
            
        label = torch.tensor(label, dtype=torch.float)
        
        return user_query, content, label

def preprocess_text(text):
    """Preprocesses text data for the model."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.split()

def create_user_profiles(interactions):
    """Creates user profiles from interaction data."""
    user_profiles = {}
    for interaction in interactions:
        user_id = interaction.get("user_id")
        if user_id not in user_profiles:
            user_profiles[user_id] = {
                "preferences": {},
                "interests": [],
                "interaction_history": []
            }
        user_profiles[user_id]["interaction_history"].append(interaction)
    return user_profiles

def load_preprocessed_data(filepath, interactions, sentiment_analyzer):
    """Loads preprocessed data from a file or processes it if the file does not exist."""
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            preprocessed_data = []
            for interaction in interactions:
                preprocessed_data.append({
                    "user_query": preprocess_text(interaction["user_query"]),
                    "agent_response": preprocess_text(interaction["response"]),
                    "time_of_day": datetime.fromtimestamp(interaction["timestamp"]).hour,
                    "user_location": "Munich, Germany",
                    "sentiment": sentiment_analyzer(interaction["user_query"])[0]["label"]
                })
            with open(filepath, "wb") as f:
                pickle.dump(preprocessed_data, f)
            return preprocessed_data
    except Exception as e:
        print(f"Error in load_preprocessed_data: {e}")
        # Return empty data as fallback
        return []

def train_model(model, dataloader, criterion, optimizer, device, epochs=10):
    """Trains the model."""
    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            # Unpack batch - batch should be a tuple of (user_query, content, label)
            if len(batch) != 3:
                print(f"Warning: Expected batch to contain 3 elements, but got {len(batch)}")
                continue
                
            user_query, content, label = batch
            
            # Move to device only if they're tensors
            if isinstance(user_query, torch.Tensor):
                user_query = user_query.to(device)
            if isinstance(content, torch.Tensor):
                content = content.to(device)
            if isinstance(label, torch.Tensor):
                label = label.to(device)

            optimizer.zero_grad()
            try:
                outputs = model(user_query)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"Error during training: {e}")
                continue

def evaluate_model(model, dataloader, device):
    """Evaluates the model and returns precision, recall, and F1 score."""
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch safely
            if len(batch) != 3:
                print(f"Warning: Expected batch to contain 3 elements, but got {len(batch)}")
                continue
                
            user_query, content, label = batch
            
            # Move to device only if they're tensors
            if isinstance(user_query, torch.Tensor):
                user_query = user_query.to(device)
            if isinstance(content, torch.Tensor):
                content = content.to(device)
            if isinstance(label, torch.Tensor):
                label = label.to(device)

            try:
                outputs = model(user_query)
                predictions = (outputs > 0.5).float()
                
                # Convert tensors to numpy for metrics calculation
                if isinstance(predictions, torch.Tensor):
                    predictions_np = predictions.cpu().numpy().flatten()
                else:
                    predictions_np = predictions
                    
                if isinstance(label, torch.Tensor):
                    labels_np = label.cpu().numpy().flatten()
                else:
                    labels_np = label
                
                all_predictions.extend(predictions_np)
                all_labels.extend(labels_np)
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue

    # Default values if we don't have any predictions
    if len(all_predictions) == 0 or len(all_labels) == 0:
        return 0.0, 0.0, 0.0
        
    try:
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return 0.0, 0.0, 0.0
        
    return precision, recall, f1

def main(model_manager, sentiment_analyzer, config):
    try:
        interactions = model_manager.get_interactions()
        
        if not interactions:
            print("Warning: No interactions found for training. Using sample data instead.")
            # Use sample data if no interactions
            interactions = [
                {"user_query": "Hello", "response": "Hi there!", "timestamp": 1609459200, "user_id": 1},
                {"user_query": "How are you?", "response": "I'm fine, thank you.", "timestamp": 1609459260, "user_id": 1}
            ]
        
        # Use the configured path for preprocessed data
        preprocessed_data_path = config.PREPROCESSED_DATA_PATH
        preprocessed_data = load_preprocessed_data(preprocessed_data_path, interactions, sentiment_analyzer)
        
        if not preprocessed_data:
            print("Error: Failed to load or create preprocessed data.")
            return
            
        user_profiles = create_user_profiles(preprocessed_data)

        content_data = [interaction["user_query"] for interaction in preprocessed_data]
        content_data += [interaction["agent_response"] for interaction in preprocessed_data]

        vocabulary = set(word for content in content_data for word in content)
        if not vocabulary:
            print("Error: Empty vocabulary. Cannot train model.")
            return
            
        word_to_index = {word: index for index, word in enumerate(vocabulary)}

        indexed_content_data = [[word_to_index[word] for word in content if word in word_to_index] for content in content_data]

        train_data, test_data = train_test_split(preprocessed_data, test_size=0.2, random_state=42)

        model = ContentRecommendationModel(input_dim=len(vocabulary))
        model.to(model_manager.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())

        # Add content field if it doesn't exist
        for item in train_data:
            if "content" not in item:
                item["content"] = item.get("agent_response", [])
                
        for item in test_data:
            if "content" not in item:
                item["content"] = item.get("agent_response", [])

        train_dataset = ContentDataset(train_data, indexed_content_data)
        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        train_model(model, train_dataloader, criterion, optimizer, model_manager.device, epochs=config.EPOCHS)

        test_dataset = ContentDataset(test_data, indexed_content_data)
        test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        precision, recall, f1 = evaluate_model(model, test_dataloader, model_manager.device)
        print(f"Content model evaluation: Precision={precision}, Recall={recall}, F1={f1}")

        # Make sure the directory exists before saving
        model_dir = os.path.dirname(config.MODEL_PATHS["content"])
        os.makedirs(model_dir, exist_ok=True)
        
        # Save using the configured path
        torch.save(model.state_dict(), config.MODEL_PATHS["content"])
        print(f"Model saved to {config.MODEL_PATHS['content']}")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        # Import required modules
        import os
        from tools.memory_manager import create_memory_manager
        
        # Initialize configuration
        config = Config()
        config.ensure_directories()  # Make sure all required directories exist
        
        # Create memory manager
        memory_manager = create_memory_manager()
        
        # Create model manager with memory manager dependency
        model_manager = ModelManager(config=config, memory_manager=memory_manager)
        
        # Create sentiment analyzer
        try:
            sentiment_analyzer = pipeline("sentiment-analysis")
        except Exception as e:
            print(f"Error creating sentiment analyzer: {e}. Using a mock instead.")
            # Mock sentiment analyzer in case of errors
            class MockSentimentAnalyzer:
                def __call__(self, text):
                    return [{"label": "POSITIVE", "score": 0.9}]
            sentiment_analyzer = MockSentimentAnalyzer()
        
        # Run main function
        print("Starting content model training...")
        main(model_manager, sentiment_analyzer, config)
        print("Content model training completed!")
    except Exception as e:
        print(f"Error in train_content_model.py: {e}")
        import traceback
        traceback.print_exc()