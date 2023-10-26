from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np

class ReviewDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels if labels is not None else [0]*len(encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    labels, reviews = [], []
    for line in lines:
        label, review = line.strip().split('\t')
        labels.append(label)
        reviews.append(review)
    return labels, reviews

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
labels, reviews = load_data('train.txt')
inputs = tokenizer(reviews, truncation=True, padding=True)

label_encoder = LabelEncoder()
labels_num = label_encoder.fit_transform(labels)

train_dataset = ReviewDataset(inputs, labels_num)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

def predict(filename):
    with open(filename, 'r') as f:
        reviews = [line.strip() for line in f.readlines()]
    inputs = tokenizer(reviews, truncation=True, padding=True)
    dataset = ReviewDataset(inputs)
    predictions = trainer.predict(dataset)
    return np.argmax(predictions.predictions, axis=-1)

predictions = predict('test_just_reviews.txt')
print(predictions)