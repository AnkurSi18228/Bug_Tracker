#import essential libraries
import re
import pickle
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


def clean_code(text):
    # remove comments
    text = re.sub(r'[^a-zA-Z0-9_\s]', ' ', text)
    #tokenize and lower case words
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)
st.title("Bug pridiction and detection system")
st.write("Enter python code snippet to check if it contains any bug.")
user_input = st.text_area("Enter your code here:")
button= st.button("Predict!")

if button:
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    model = pickle.load(open('rf_model.pkl', 'rb'))
    # clean the input code
    cleaned_code = clean_code(user_input)
    # vectorize the cleaned code
    input_tfidf = vectorizer.transform([cleaned_code])
    # predict the bug status
    prediction = model.predict(input_tfidf)[0]
    # display the result
    if prediction:
        st.error("The code contains a bug.")
    else:
        st.success("The code does not contain any bug.")
        
#run streamlit run main.py to run the app