import spacy

# Load the spaCy English language model
nlp = spacy.load('en_core_web_sm')

# Define a list of positive and negative relationship keywords
positive_keywords = ['friend', 'buddy', 'pal', 'mate']
negative_keywords = ['enemy', 'rival', 'foe']

# Function to analyze conversations and infer social relationships
def analyze_conversations(conversations):
    for conversation in conversations:
        doc = nlp(conversation)

        # Check if any positive relationship keyword is present
        for token in doc:
            if token.text.lower() in positive_keywords:
                return 'Positive relationship'

        # Check if any negative relationship keyword is present
        for token in doc:
            if token.text.lower() in negative_keywords:
                return 'Negative relationship'

    # If no relationship keywords are found, return 'Unknown'
    return 'Unknown'

# Example conversations
conversations = [
    "Hey, let's grab lunch together!",
    "I can't stand that person. We are always competing.",
    "We've known each other since childhood. We're good friends.",
    "I can't believe he did that. He's such a jerk!"
]

# Analyze the conversations
relationship = analyze_conversations(conversations)

# Output the inferred relationship
print(f"Inferred Relationship: {relationship}")
