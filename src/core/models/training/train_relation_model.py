from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from models.model_manager import ModelManager
import time
from config.config import Config

def preprocess_text(text):
    """Preprocesses text data for the model."""
    text = text.lower().strip()
    # Add more preprocessing steps as needed
    return text

def prepare_datasets(preprocessed_data):
    """Prepares datasets for training and evaluation."""
    texts = [preprocess_text(data["user_query"]) for data in preprocessed_data]
    labels = [data["label"] for data in preprocessed_data]
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    return DatasetDict({"train": train_dataset, "test": test_dataset})

def initialize_model_and_tokenizer(model_name, num_labels):
    """Initializes the tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

def tokenize_function(examples, tokenizer):
    """Tokenizes the dataset examples."""
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(p):
    """Computes accuracy and classification report."""
    preds = p.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "classification_report": classification_report(p.label_ids, preds, output_dict=True)
    }

def train_and_evaluate(model, tokenizer, datasets, training_args):
    """Trains and evaluates the model."""
    tokenized_datasets = datasets.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics
    )
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    trainer.save_model(Config.RELATION_MODEL_PATH)
    tokenizer.save_pretrained(Config.RELATION_MODEL_PATH)

def main(model_manager, config):
    model_manager.add_interaction({
        "user_query": "How is the weather today?",
        "response": "It's sunny.",
        "timestamp": time.time(),
        "user_location": "New York",
        "sentiment": "positive"
    })
    preprocessed_data = model_manager.fetch_and_preprocess_interaction_data_for_relation()
    datasets = prepare_datasets(preprocessed_data)
    model_name = "bert-base-uncased"
    tokenizer, model = initialize_model_and_tokenizer(model_name, num_labels=len(set(datasets["train"]["label"])))
    training_args = TrainingArguments(
        output_dir=config.TRAINING_OUTPUT_DIR,
        num_train_epochs=config.TRAINING_NUM_EPOCHS,
        per_device_train_batch_size=config.TRAINING_BATCH_SIZE,
        per_device_eval_batch_size=config.TRAINING_BATCH_SIZE * 4,
        warmup_steps=config.TRAINING_WARMUP_STEPS,
        weight_decay=config.TRAINING_WEIGHT_DECAY,
        logging_dir=config.TRAINING_LOGGING_DIR,
        logging_steps=config.TRAINING_LOGGING_STEPS,
        evaluation_strategy=config.TRAINING_EVALUATION_STRATEGY
    )
    train_and_evaluate(model, tokenizer, datasets, training_args)

if __name__ == "__main__":
    model_manager = ModelManager(auto_update=False)
    config = Config()
    main(model_manager, config)
