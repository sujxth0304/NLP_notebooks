import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import docx2txt
import PyPDF2
import re

# For reading PDF files
def read_pdf(file):
    pdf_reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page).extract_text()
    return text

# For reading DOCX files
def read_docx(file):
    return docx2txt.process(file)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

st.title("Resume Matcher")

uploaded_files = st.file_uploader("Upload Resumes", type=['pdf', 'docx'], accept_multiple_files=True)
job_description = st.text_area("Enter Job Description")

if st.button("Process"):
    if uploaded_files and job_description:
        resume_texts = []
        resume_names = []

        for file in uploaded_files:
            if file.name.endswith('.pdf'):
                resume_texts.append(read_pdf(file))
            elif file.name.endswith('.docx'):
                resume_texts.append(read_docx(file))
            resume_names.append(file.name)

        # Preprocess resumes
        resume_texts = [preprocess_text(text) for text in resume_texts]
        
        # Create a DataFrame
        df = pd.DataFrame({'Resume': resume_texts, 'File Name': resume_names})
        
        # Preprocess and feature extraction
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df['Resume'])
        
        # Simulate labels for training (assuming matching resumes are labeled as 1 and non-matching as 0)
        # In a real scenario, you would need a labeled dataset
        y = np.random.randint(0, 2, size=X.shape[0])
        
        # Train-test split
        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X, y, df['File Name'], test_size=0.4, random_state=42)
        
        # Train logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Predict job description similarity
        job_desc_vector = vectorizer.transform([preprocess_text(job_description)])
        job_desc_prediction = model.predict(X_test)

        # Get matching resumes
        matched_resumes = names_test[job_desc_prediction == 1]
        
        # Display results
        st.write("Matched Resumes:")
        if len(matched_resumes) > 0:
            for resume in matched_resumes:
                st.write(resume)
        else:
            st.write("No matching resumes found.")
    else:
        st.warning("Please upload resumes and enter a job description.")
