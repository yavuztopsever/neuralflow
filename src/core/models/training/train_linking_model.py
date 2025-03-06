import re
import pickle
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from models.model_manager import ModelManager
from config.config import Config

def preprocess_text(text):
    """Preprocesses text data for the model."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fine_tune_model(model, tokenizer, train_dataset, val_dataset, model_save_path):
    """Fine-tunes the model with the given datasets and saves the model."""
    training_args = TrainingArguments(**Config.TRAINING_ARGS)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    # Save the model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

def main(model_manager, tokenizer, model):
    """Main function to run the training and inference."""
    # Fetch and preprocess interaction data for linking model
    preprocessed_data = model_manager.fetch_and_preprocess_interaction_data_for_linking()

    # Save preprocessed data for training
    with open("training/preprocessed_linking_data.pkl", "wb") as f:
        pickle.dump(preprocessed_data, f)

    # Example text
    text = "Hugging Face Inc. is a company based in New York City."

    # Preprocess text
    preprocessed_text = preprocess_text(text)

    # Perform NER using ModelManager
    entities = model_manager.infer_linking(preprocessed_text)
    print(entities)

    # Fine-tune the model with new data
    train_dataset = ...  # Load or create your training dataset
    val_dataset = ...    # Load or create your validation dataset

    fine_tune_model(model, tokenizer, train_dataset, val_dataset, Config.MODEL_SAVE_PATH)
    # No need to reload the model and tokenizer as they are already updated
    assert hasattr(model, 'state')  # Ensure state attribute is present

if __name__ == "__main__":
    # Initialize ModelManager
    model_manager = ModelManager(auto_update=False)
    
    # Load a pre-trained model for named entity recognition (NER)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(Config.MODEL_NAME)
    
    main(model_manager, tokenizer, model)