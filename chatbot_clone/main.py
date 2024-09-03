import streamlit as st
from transformers import pipeline
import pandas as pd

# Initialize the question-answering pipeline using a pretrained model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Sample dataset of questions and answers
data = {
    "Question": [
        "What is Rhapsody?",
        "How do I create a new project in Rhapsody?",
        "What are the benefits of using Rhapsody?",
        "How to generate code from a model in Rhapsody?"
    ],
    "Answer": [
        "Rhapsody is a visual modeling tool for designing complex systems.",
        "To create a new project in Rhapsody, go to File > New Project and follow the wizard.",
        "The benefits of using Rhapsody include model-based design, collaboration features, and code generation.",
        "To generate code, click on Tools > Generate Code and select the model elements."
    ]
}

# Load the dataset into a DataFrame
df = pd.DataFrame(data)

# Create a context from your dataset
context = " ".join(df["Answer"])

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
