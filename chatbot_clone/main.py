import streamlit as st
from transformers import pipeline
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Get the absolute path of the current script
current_path = os.path.abspath(__file__)




# Initialize the question-answering pipeline using a pretrained model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Sample dataset of questions and answers
with open('/mount/src/chatbot/chatbot_clone/q.json') as f:
  data = json.load(f)


# Load the dataset into a DataFrame
df = pd.DataFrame(data["data"])

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
            st.write(f"{current_path}")
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
