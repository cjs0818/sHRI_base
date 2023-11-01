
#---------------------------------------------
# https://blog.naver.com/oioio11/223110858617
#
# pip install openai


import os
import openai
#from dotenv import load_dotenv
#load_dotenv('../.env.local')

openai.api_key = os.getenv("OPENAI_API_KEY")
model_engine = "gpt-3.5-turbo"
#input_text = "해바라기 꽃말에 대해 알려줘"
input_text = "오늘 날씨 어때?"

response = openai.ChatCompletion.create(
   model=model_engine,
   messages=[{"role": "user", "content": input_text }]
)

output_text = response['choices'][0]['message']['content']
print("[Bot]: ", output_text)

'''
from transformers import pipeline

# Define the pipeline
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.38')

# Define the conversation history and generate reponse
conversation_history = ""
while True:
	user_input = input(">> User: ")
	conversation_history += f"\nUser: {user_input}"
	response = generator(conversation_history, max_length=30, num_return_sequences=1)
	conversation_history += f"\nBot: {response}"
	print(f"Bot: {response}")
'''