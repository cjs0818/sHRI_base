from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Function to classify the relationship as senior or junior
def classify_relationship(conversations):
    inputs = tokenizer(conversations, padding=True, truncation=True, return_tensors='pt')
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_labels = torch.argmax(logits, dim=1)
    labels = ['junior', 'senior']
    predicted_relationship = labels[predicted_labels.item()]
    return predicted_relationship

# Example conversations
conversations = "Hey, can you show me how to do this? I'm new here.", "Sure, I'd be happy to help. I've been working on this project for a year."

# Classify the relationship
relationship = classify_relationship(conversations)

# Output the predicted relationship
print(f"Predicted Relationship: {relationship}")
