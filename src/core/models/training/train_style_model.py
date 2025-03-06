from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import pickle
from sklearn.metrics import accuracy_score, classification_report
from config.config import Config  # Import Config

def preprocess_text(text):
    """Preprocess text data for the model."""
    return text.lower().strip()

def compute_metrics(pred):
    """Compute metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True)
    return {"accuracy": accuracy, "classification_report": report}

def load_data(filepath):
    """Load preprocessed data from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)

def create_datasets(texts, labels, test_size=0.2):
    """Create train and test datasets."""
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=test_size)
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    return DatasetDict({"train": train_dataset, "test": test_dataset})

def tokenize_function(examples, tokenizer):
    """Tokenize the datasets."""
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def main(config):
    """Main function to train and evaluate the model."""
    # Load and preprocess data
    style_data = load_data(config.PREPROCESSED_STYLE_DATA_PATH)
    texts, labels = zip(*style_data)
    texts = [preprocess_text(text) for text in texts]

    # Create datasets
    datasets = create_datasets(texts, labels)

    # Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=len(set(labels)))

    # Tokenize datasets
    tokenized_datasets = datasets.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config.TRAINING_OUTPUT_DIR,
        num_train_epochs=config.TRAINING_NUM_EPOCHS,
        per_device_train_batch_size=config.TRAINING_BATCH_SIZE,
        per_device_eval_batch_size=config.TRAINING_BATCH_SIZE,
        warmup_steps=config.TRAINING_WARMUP_STEPS,
        weight_decay=config.TRAINING_WEIGHT_DECAY,
        logging_dir=config.TRAINING_LOGGING_DIR,
        logging_steps=config.TRAINING_LOGGING_STEPS,
        evaluation_strategy=config.TRAINING_EVALUATION_STRATEGY
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics
    )

    # Train and evaluate the model
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the trained model
    trainer.save_model(config.MODEL_PATHS["style"])

if __name__ == "__main__":
    config = Config()  # Create a Config instance
    main(config)
