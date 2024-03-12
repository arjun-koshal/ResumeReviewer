import streamlit as st
import fitz
import spacy
import pandas as pd
import docx2txt
import re
import nltk

# Load the SpaCy NER model
nlp_ner = spacy.load("./model")

# Load the resume dataset
resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')

# Load NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Clean resume text
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub('#\S+', '', resumeText)
    resumeText = re.sub('@\S+', '  ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

# Streamlit App
st.title("Resume Parser and Category Classifier")

# File upload widget for multiple files
uploaded_files = st.file_uploader("Upload Multiple Files (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # Read the uploaded file
            file_extension = uploaded_file.name.split('.')[-1]

            if file_extension == 'pdf':
                # Read the PDF file
                pdf_data = uploaded_file.read()
                doc = fitz.open("uploaded.pdf", pdf_data)
                text = " ".join([page.get_text() for page in doc])

            elif file_extension == 'docx':
                # Read the DOCX file
                text = docx2txt.process(uploaded_file)

            # Process the text with the SpaCy NER model
            parsed_doc = nlp_ner(text)

            # Display the extracted entities
            st.header("Extracted Entities from Resume:")
            for ent in parsed_doc.ents:
                st.write(f"{ent.label_} : {ent.text}")

            st.divider()

            # Create a list of entities and labels
            entities = [ent.text for ent in parsed_doc.ents]
            labels = [ent.label_ for ent in parsed_doc.ents]

            # Create a Pandas DataFrame
            data = {'Label': labels, 'Content': entities}
            df = pd.DataFrame(data)

            # Display the extracted entities in a table
            st.header("Extracted Entities in Tabular format:")
            st.dataframe(df)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")