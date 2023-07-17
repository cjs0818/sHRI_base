from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# You need to "git clone https://github.com/e9t/nsmc.git"
import numpy as np
import pandas as pd

BASE_PATH='/home/jschoi/work/sHRI_base/conversation/practice/'
MODEL_PATH = BASE_PATH + 'weights/'


#-------------------------------------------------------
# Read data from csv file
train_df = pd.read_csv(BASE_PATH + 'nsmc/ratings_train.txt', sep='\t')
test_df = pd.read_csv(BASE_PATH + 'nsmc/ratings_test.txt', sep='\t')
#train_df.drop(['id'], axis=1, inplace=True, index=False)

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

train_df = train_df.sample(frac=0.4, random_state=999)
test_df = test_df.sample(frac=0.4, random_state=999)
#train_df = train_df.sample(frac=0.01, random_state=999)
#test_df = test_df.sample(frac=0.01, random_state=999)
#-------------------------------------------------------


state = 0 # 0: train, 1: test

#-------------------------------------------------------
# Prepare your dataset
#conversations = ["Hey, can you show me how to do this? I'm new here.", "Sure, I'd be happy to help. I've been working on this project for a year."] 
#labels = [0, 1]  # List of labels (e.g., 0 for "junior", 1 for "senior")

conversations = train_df['document'].values
labels = train_df['label'].values
#-------------------------------------------------------


# Define your dataset class
class MyDataset(Dataset):
    def __init__(self, conversations, labels, tokenizer):
        self.conversations = conversations
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        encoded_input = self.tokenizer.encode_plus(
            self.conversations[idx],
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )


        input_ids = encoded_input['input_ids'].squeeze()
        attention_mask = encoded_input['attention_mask'].squeeze()

        #print(f"self.labels: {self.labels}")
        #print(f"self.labels[idx]: {self.labels[idx]}")

        label = torch.tensor(self.labels[idx])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }



# Split dataset into training and validation sets
train_conversations, val_conversations, train_labels, val_labels = train_test_split(
    conversations, labels, test_size=0.2, random_state=42
)
print(f"train_conversations: {train_conversations}")
print(f"train_labels: {train_labels}")

print(f"val_conversations: {val_conversations}")
print(f"val_labels: {val_labels}")



#-------------------------------------------------------
# Load pre-trained BERT model and tokenizer

#model_name = 'bert-base-uncased'
model_name = 'beomi/kcbert-base'    # Korean BERT model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
#-------------------------------------------------------


# Create data loaders for training and validation sets
train_dataset = MyDataset(train_conversations, train_labels, tokenizer)
val_dataset = MyDataset(val_conversations, val_labels, tokenizer)

#print(f"train_dataset.conversations: {train_dataset.conversations}")
#print(f"train_dataset.labels: {train_dataset.labels}")
#print(train_dataset[0]['input_ids'])

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define training parameters
num_epochs = 10
learning_rate = 2e-6
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()


#-------------------------------------------------------
# Load model
class CustomModel(torch.nn.Module):
    pass

#model = CustomModel().to(device)

#torch.load(MODEL_PATH + 'model.pt', map_location=device)

#model_state_dict = torch.load(MODEL_PATH + 'model_state_dict.pt', map_location=device)
#model.load_state_dict(model_state_dict)

#checkpoint = torch.load(MODEL_PATH + 'all.tar')
#model.load_state_dict(checkpoint['model'])

#optimizer.load_state_dict(checkpoint['optimizer'])
#-------------------------------------------------------


if state == 0:  # 0: train, 1: test

    # Training loop
    checkpoint = 1
    for epoch in range(num_epochs):
        model.train()

        for batch in train_loader:
            #print(f"batch: {batch['input_ids']}")

            #print("batch:", batch)
            #print("input_ids: {}\n label: {}".format(batch['input_ids'], batch['labels']))


            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                _, predicted = torch.max(outputs.logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Print validation metrics
        val_loss /= len(val_loader)
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}: Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        if (epoch + 1) % 2 == 0: # at every second epoch
            #-------------------------------------------------------
            # Save model
            #torch.save(model, MODEL_PATH + 'model.pt')
            #torch.save(model.state_dict(), MODEL_PATH + 'model_state_dict.pt')
            torch.save({
                "model": "CustomModel",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "description": f"CustomModel checkpoint-{checkpoint}"
            }, MODEL_PATH + f"checkpoint-{checkpoint}.pt")
            #-------------------------------------------------------
            checkpoint += 1

elif state == 1:
    print(state)