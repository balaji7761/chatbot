import streamlit as st
from transformers import pipeline
import pandas as pd
import json

# Initialize the question-answering pipeline using a pretrained model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Define the dataset
with open('q.json') as f:
    data = json.load(f)


# Extract data from the dataset
data_list = data["data"]

# Load the dataset into a DataFrame
df = pd.DataFrame(data_list)

# Create a context from your dataset
context = " ".join(df["answer"])

# Function to answer questions using the pretrained model
def answer_question(question, context):
    try:
        result = qa_pipeline({"question": question, "context": context})
        return result['answer']
    except Exception as e:
        return "I'm not sure how to answer that."

# Streamlit App Interface
st.title('Rhapsody Chatbot')
user_input = st.text_input("Ask a question about Rhapsody:")

# Handle different types of user input
if user_input:
    if user_input.lower() in ["hi", "hello", "hey"]:
        st.write("Hello! How can I help you with Rhapsody today?")
    else:
        response = answer_question(user_input, context)
        st.write(response)
