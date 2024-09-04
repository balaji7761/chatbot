import streamlit as st
from transformers import pipeline
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
# Get the absolute path of the current script
current_path = os.path.abspath(__file__)




# Initialize the question-answering pipeline using a pretrained model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Step 1: Read the content from the text file
try:
    with open('/mount/src/chatbot/chatbot_clone/q.txt', 'r') as file:
        # Read the entire content of the file as a string
        content = file.read()
        print("File content read successfully.")

    # Step 2: Convert the content to a JSON object
    try:
        data = json.loads(content)  # Parses the string to a JSON object (dict or list)
        print("JSON data loaded successfully.")

        # Step 3: Extract "question" and "answer" into x and y variables
        if "data" in data and isinstance(data["data"], list):
            # Extract questions and answers into separate lists
            x = [item["question"] for item in data["data"]]
            y = [item["answer"] for item in data["data"]]

            # Step 4: Create a DataFrame with 'question' as x and 'answer' as y
            df = pd.DataFrame({"question": x, "answer": y})
            print("DataFrame created successfully:")
            print(df)
        else:
            print("Error: The key 'data' is not present or not formatted correctly.")
    except json.JSONDecodeError as e:
        print("Error: Failed to decode JSON. Please check the content of the file:", e)

except FileNotFoundError:
    print("Error: The specified file was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")



# Load the dataset into a DataFrame
# df = pd.DataFrame(data["data"])

# Create a context from your dataset
context = " ".join(df["answer"])

# Vectorize the questions for similarity check
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])

# Function to find the most similar question above a similarity threshold
def get_most_similar_answer(user_question, threshold=0.7):
    user_question_vec = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_question_vec, X).flatten()
    # Find the index of the most similar question
    best_match_index = similarities.argmax()
    best_similarity = similarities[best_match_index]
    # Return the answer if the similarity is above the threshold
    if best_similarity >= threshold:
        return df["answer"].iloc[best_match_index], df["question"].iloc[best_match_index], best_similarity
    return None, None, best_similarity

# Function to find similar questions to the user's input
def find_similar_questions(user_question, top_n=3):
    user_question_vec = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_question_vec, X).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(df["question"].iloc[i], df["answer"].iloc[i]) for i in top_indices if similarities[i] > 0]

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
        # Check for a similar question above the threshold
        dataset_answer, matched_question, similarity = get_most_similar_answer(user_input, threshold=0.5)
        
        if dataset_answer:
            
            response = dataset_answer
        else:
            response = "Ask question related to Rhapsody"
        st.write(f"Answer: {response}")

        # Asking if the user needs suggestion questions
        suggest = st.radio("Do you need any suggestion questions?", ("No", "Yes"))

        if suggest == "Yes":
            similar_questions = find_similar_questions(user_input)
            question_choices = [q for q, _ in similar_questions]
            if question_choices:
                selected_question = st.selectbox("Select a question:", question_choices)

                if selected_question:
                    # Find the answer corresponding to the selected question
                    answer = dict(similar_questions)[selected_question]
                    st.write(f"Answer: {answer}")
            else:
                st.write("No similar questions found.")
