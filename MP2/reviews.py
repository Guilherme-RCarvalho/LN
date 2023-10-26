from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import torch

def read_training_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            label, review = line.strip().split("\t")
            data.append((review, label))
    return data

def read_test_data(file_path):
    test_reviews = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            review = line.strip()
            test_reviews.append(review)
    return test_reviews

def preprocess_data(train_data, test_data):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    label_encoder = LabelEncoder()

    # Split train_data into reviews and labels
    train_reviews, train_labels = zip(*train_data)

    # Tokenize the input data using DistilBERT tokenizer
    train_encodings = tokenizer(list(train_reviews), truncation=True, padding=True)
    test_encodings = tokenizer(test_data, truncation=True, padding=True)

    # Convert tokenized input into PyTorch tensors
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        torch.tensor(label_encoder.fit_transform(train_labels))  # Encode labels
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask'])
    )
    print(tokenizer.decode(train_dataset[0][0]))  # Decode the first tokenized input from the training dataset
    print(tokenizer.decode(test_dataset[0][0]))   # Decode the first tokenized input from the test dataset

    return train_dataset, test_dataset, label_encoder

def train_model(train_dataset, training_args):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=lambda p: {'accuracy': (p.argmax(axis=1) == val_labels).mean()}
    )
    trainer.train()
    return model

def predict_labels(model, test_dataset):
    predictions = trainer.predict(test_dataset)
    predicted_labels = label_encoder.inverse_transform(predictions.predictions.argmax(axis=1))
    return predicted_labels

def main():
    # Read training data
    train_data = read_training_data("train.txt")
    
    # Read test data
    test_data = read_test_data("test_just_reviews.txt")
    
    # Preprocess data
    train_dataset, test_dataset, label_encoder = preprocess_data(train_data, test_data)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    print(train_dataset[:5])  # Print the first 5 elements of the training dataset
    print(test_dataset[:5])   # Print the first 5 elements of the test dataset
    
    # Train the model
    trained_model = train_model(train_dataset, training_args)
    
    # Predict labels for test data
    predicted_labels = predict_labels(trained_model, test_dataset)
    
    # Print or save predicted_labels as needed
    print(predicted_labels)

if __name__ == "__main__":
    main()
