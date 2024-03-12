import streamlit as st
import en_core_web_sm
import fitz
from scipy import spatial
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from io import BytesIO
import base64
import pandas as pd
import docx2txt

nlp = en_core_web_sm.load()

st.title("ResumeRank: Scoring Candidate's Employment Match")

# Functions to extract text, extract keywords, etc.
def extract_text_from_stream(stream):
    with fitz.open(stream=stream) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_keywords(text):
    """
    Extracts keywords from text, removing stopwords and non-alphanumeric characters.
    """
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return set(filtered_tokens)

def create_keywords_vectors(keyword, nlp):
    """Converts a keyword to its vector representation."""
    doc = nlp(keyword)
    return doc.vector

def cosine_similarity(vect1, vect2):
    """Calculates the cosine similarity between two vectors."""
    return 1 - spatial.distance.cosine(vect1, vect2)

def get_similar_words(keywords, nlp):
    """Finds similar words for a list of keywords."""
    extended_keywords = set()
    for keyword in keywords:
        keyword_vector = create_keywords_vectors(keyword, nlp)
        similarity_list = []
        for token in nlp.vocab:
            if token.has_vector and token.is_lower and token.is_alpha:
                similarity = cosine_similarity(keyword_vector, token.vector)
                if similarity > 0.5:  # Threshold for similarity
                    similarity_list.append((token.text, similarity))

    return list(extended_keywords)

def highlight_in_stream(stream, keywords, extended_keywords):
    # Convert the BytesIO stream to a PDF document object
    doc = fitz.open(stream=stream)
    degree_patterns = ['B.S.', 'B.A.', 'B.E.', 'Bachelors of Arts', 'Bachelors of Science', 'Bachelors of Engineering',
                       'Bachelor of Arts', 'Bachelor of Science', 'Bachelor of Engineering', 'Dual',
                       "Bachelor's of Arts", "Bachelor's of Science", "Bachelor's of Engineering",
                       'Masters of Arts', 'Masters of Science', 'Masters of Engineering', "M.A.", "M.S.", "M.E.",
                       "Master's of Arts", "Master's of Science", "Master's of Engineering", "Ph.D.", "Minor", "Concentration"]

    extended_keywords = get_similar_words(keywords, nlp)
    all_keywords = set(keywords).union(extended_keywords)

    for page in doc:
        for keyword in all_keywords:
            for inst in page.search_for(keyword, quads=True):
                annot = page.add_highlight_annot(inst)
                annot.set_colors(stroke=[1, 1, 0])  # Yellow for keywords and related terms
                annot.update()

        # Highlight degree patterns in green
        for pattern in degree_patterns:
            for inst in page.search_for(pattern, quads=True):
                annot = page.add_highlight_annot(inst)
                annot.set_colors(stroke=[0, 1, 0])  # Green for degree names/types
                annot.update()

    # Return the modified PDF as a stream
    pdf_stream = BytesIO()
    doc.save(pdf_stream, garbage=4, deflate=True)
    doc.close()
    pdf_stream.seek(0)
    return pdf_stream

# UI components for file upload
jd_file = st.file_uploader("Upload the job description (PDF, DOCX)", type=["pdf", "docx"])
resume_file = st.file_uploader("Upload the resume (PDF, DOCX)", type=["pdf", "docx"])

if st.button("Match Resume to Job"):
    if jd_file is not None and resume_file is not None:
        # Convert uploaded files to BytesIO streams
        jd_stream = BytesIO(jd_file.getvalue())
        resume_stream = BytesIO(resume_file.getvalue())

        # Extract text from job description
        jd_text = extract_text_from_stream(jd_stream)
        # Extract keywords from the job description text
        keywords = extract_keywords(jd_text)

        # Highlight keywords in the resume PDF and get the modified stream
        highlighted_pdf_stream = highlight_in_stream(resume_stream, keywords, get_similar_words(keywords, nlp))

        # Convert the modified PDF stream to base64 for display
        base64_pdf = base64.b64encode(highlighted_pdf_stream.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.error("Please upload both job description and resume PDFs.")

if resume_file is not None:
    # Read the uploaded file
    file_extension = resume_file.name.split('.')[-1]

    if file_extension == 'pdf':
        # Read the PDF file
        pdf_data = resume_file.read()
        doc = fitz.open("uploaded.pdf", pdf_data)
        text = " ".join([page.get_text() for page in doc])

    elif file_extension == 'docx':
        # Read the DOCX file
        text = docx2txt.process(resume_file)

    # Process the text with the SpaCy NER model
    parsed_doc = nlp(text)

    # Display the extracted entities
    st.header("Extracted Entities:")
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