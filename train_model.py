import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split

# Load the cleaned data
df = pd.read_csv('cleaned_data.csv')

# Extract message and labels
messages = df['cleaned_message'].values
labels = df['label'].values

# Step 2: Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

# Tokenize data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

X_train = [str(message) for message in X_train]
X_test = [str(message) for message in X_test]

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# Custom Dataset Class for PyTorch
class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SpamDataset(train_encodings, y_train)
test_dataset = SpamDataset(test_encodings, y_test)

# Training args
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Load the DistilBERT model for classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    tokenizer=tokenizer
)

# Train the Model
print("Starting model training...")
trainer.train()

# Evaluate the Model
print("Evaluating the model...")
results = trainer.evaluate()
print(f"Evaluation results: {results}")

# Save the trained model and tokenizer locally
print("Saving the model...")
model.save_pretrained('scam_detector_model')
tokenizer.save_pretrained('scam_detector_model')