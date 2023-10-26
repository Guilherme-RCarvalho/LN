from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class ReviewDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels if labels is not None else [0]*len(encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Add dtype=torch.long
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    labels, reviews = [], []
    for line in lines:
        label, review = line.strip().split('\t')
        labels.append(label)
        reviews.append(review)
    return labels, reviews

# Initialize lemmatizer and get English stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize and lowercase
    words = word_tokenize(text.lower())
    # Remove punctuation and stopwords, and lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
labels, reviews = load_data('train.txt')
reviews = [preprocess_text(review) for review in reviews]
inputs = tokenizer(reviews, truncation=True, padding=True)

label_encoder = LabelEncoder()
labels_num = label_encoder.fit_transform(labels)

#train_dataset = ReviewDataset(inputs, labels_num)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)


#train
train_reviews, val_reviews, train_labels, val_labels = train_test_split(reviews, labels, test_size=0.1)


train_inputs = tokenizer(train_reviews, truncation=True, padding=True)
val_inputs = tokenizer(val_reviews, truncation=True, padding=True)

train_labels_num = label_encoder.transform(train_labels)
val_labels_num = label_encoder.transform(val_labels)


train_dataset = ReviewDataset(train_inputs, train_labels_num)
val_dataset = ReviewDataset(val_inputs, val_labels_num) 

def objective(trial):
    # Define hyperparameters using trial object
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    
    # Use hyperparameters in your Trainer
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=lr,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Evaluate model and return validation accuracy
    eval_result = trainer.evaluate(eval_dataset=val_dataset)
    return eval_result['eval_accuracy']

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_params)
#training_args = TrainingArguments(
#    output_dir='./results',
#    num_train_epochs=1,
#    per_device_train_batch_size=16,
#    per_device_eval_batch_size=64,
#    warmup_steps=500,
#    weight_decay=0.01,
#)
#
#trainer = Trainer(
#    model=model,
#    args=training_args,
#    train_dataset=train_dataset,
#    compute_metrics=compute_metrics,
#)
#
#trainer.train()
#
'''
eval_result = trainer.evaluate(eval_dataset=val_dataset)

print(f"Validation accuracy: {eval_result['eval_accuracy']}")

def predict(filename):
    with open(filename, 'r') as f:
        reviews = [line.strip() for line in f.readlines()]
    inputs = tokenizer(reviews, truncation=True, padding=True)
    dataset = ReviewDataset(inputs)
    predictions = trainer.predict(dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=-1)

    # Convert numeric labels back to original labels
    predicted_labels_str = label_encoder.inverse_transform(predicted_labels)

    # Write predictions to a file
    with open('predicted_labels.txt', 'w') as f:
        for label in predicted_labels_str:
            f.write(f"{label}\n")
'''
#predict('test_just_reviews.txt')