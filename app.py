import streamlit as st
import pickle

# Load the model
with open("trained_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Load the TfidfVectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)


# Function to predict sentiment
def predict_sentiment(review):
    # Transform input text using TfidfVectorizer
    transformed_review = tfidf_vectorizer.transform([review])

    # Make prediction
    prediction = loaded_model.predict(transformed_review)[0]

    # Return prediction result
    return prediction


# Streamlit UI
def main():
    st.title("Movie Review Sentiment Analysis")

    # Input text area for user to enter review
    review = st.text_area("Enter your movie review:")

    # Predict button
    if st.button("Predict"):
        if review:
            prediction = predict_sentiment(review)
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.success(f"The sentiment of the review is {sentiment}")


# Main function to run the Streamlit app
if __name__ == "__main__":
    main()
