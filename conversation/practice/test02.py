# test02
# You need to "git clone git clone https://github.com/e9t/nsmc.git"
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

import torch.nn.functional as F

train_df = pd.read_csv('./nsmc/ratings_train.txt', sep='\t')
test_df = pd.read_csv('./nsmc/ratings_test.txt', sep='\t')

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

train_df = train_df.sample(frac=0.4, random_state=999)
test_df = test_df.sample(frac=0.4, random_state=999)

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
        label = torch.tensor(self.labels[idx])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

#conversations = train_df['document']
#labels = train_df['label']

conversations = ["아 더빙.. 진짜 짜증나네요 목소리", "흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나", "너무재밓었다그래서보는것을추천한다", "교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정", "사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다", "막 걸음마 뗀 3세부터 초등학교 1학년생인 8살용영화.ㅋㅋㅋ...별반개도 아까움.", "원작의 긴장감을 제대로 살려내지못했다.", "별 반개도 아깝다 욕나온다 이응경 길용우 연기생활이몇년인지..정말 발로해도 그것보단 낫겟다 납치.감금만반복반복..이드라마는 가족도없다 연기못하는사람만모엿네", "액션이 없는데도 재미 있는 몇안되는 영화", "왜케 평점이 낮은건데? 꽤 볼만한데.. 헐리우드식 화려함에만 너무 길들여져 있나?"]
labels = [0, 1, 0, 0, 1, 0, 0, 0, 1, 1]

# Split dataset into training and validation sets
train_conversations, val_conversations, train_labels, val_labels = train_test_split(
    conversations, labels, test_size=0.2, random_state=42
)


# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

#model_name = 'bert-base-uncased'
#tokenizer = BertTokenizer.from_pretrained(model_name)
#model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Create data loaders for training and validation sets
train_dataset = MyDataset(train_conversations, train_labels, tokenizer)
val_dataset = MyDataset(val_conversations, val_labels, tokenizer)

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# Define training parameters
num_epochs = 1 #10
learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()


print(train_loader)


# Training loop
for epoch in range(num_epochs):
    model.train()

    for batch in train_loader:
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
