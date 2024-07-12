import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import docx2txt
import PyPDF2
import os

# Function to read PDF files
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page in range(reader.numPages):
            text += reader.getPage(page).extract_text()
        return text

# Function to read DOCX files
def read_docx(file_path):
    return docx2txt.process(file_path)

# Directory containing resumes (PDF and DOCX files)
resumes_directory = 'src/resumes'

# List to store resume text and file names
resume_texts = []
resume_names = []

# Read resumes from the directory
for file_name in os.listdir(resumes_directory):
    if file_name.endswith('.pdf'):
        file_path = os.path.join(resumes_directory, file_name)
        text = read_pdf(file_path)
    elif file_name.endswith('.docx'):
        file_path = os.path.join(resumes_directory, file_name)
        text = read_docx(file_path)
    else:
        continue
    
    # Append to lists
    resume_texts.append(text)
    resume_names.append(file_name)

# Create a DataFrame
df = pd.DataFrame({'Resume': resume_texts, 'File Name': resume_names})

# Assuming you have labels (0 for non-matches, 1 for matches)
# Replace this with your actual labeled data
# For demonstration, assuming all are positive matches (replace with your actual labels)
y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0])

# Preprocess text (you can add more preprocessing steps as needed)
def preprocess_text(text):
    # Example: Convert to lowercase
    return text.lower()

df['Resume'] = df['Resume'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Resume'])

# Train-test split
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X, y, df['File Name'], test_size=0.25, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Example of predicting with a job description
job_description = "Seeking an NLP Engineer proficient in Python and familiar with libraries such as NLTK, spaCy, and Transformers. Must have experience in developing and deploying NLP models for tasks like text classification, named entity recognition (NER), and sentiment analysis. Strong understanding of machine learning techniques, particularly in the context of natural language processing, is essential. Ability to preprocess large text corpora and optimize model performance is a plus."
job_desc_vector = vectorizer.transform([preprocess_text(job_description)])
job_desc_prediction = model.predict(job_desc_vector)

# Print out the predictions to debug
print("Predictions for the job description:", job_desc_prediction)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model performance (optional)
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Get matching resumes
matched_resumes = names_test[y_pred == 1].tolist()

# Display matched resumes
print("Matched Resumes:")
for resume in matched_resumes:
    print(resume)
