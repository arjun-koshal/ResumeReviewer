import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
import base64
import re
import os
import zipfile
from pathlib import Path
import tempfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import en_core_web_sm
import plotly.graph_objects as go

# Load NLP model
nlp = en_core_web_sm.load()

st.title("Resume Ranker: Scoring Candidate's Employment Match")

# Setup for job description selection
job_descriptions_folder = "./Job Descriptions"
job_descriptions_files = [f for f in os.listdir(job_descriptions_folder) if f.endswith('.pdf')]
job_descriptions_titles = [os.path.splitext(f)[0] for f in job_descriptions_files]

selected_job_title = st.selectbox("Select the job you're interested in:", job_descriptions_titles)
selected_job_file = os.path.join(job_descriptions_folder,
                                 next((f for f in job_descriptions_files if f.startswith(selected_job_title + '.pdf')),
                                      None))


# Function to extract and clean text from a PDF file
def extract_and_clean_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        text = " ".join(page.get_text() for page in doc)
    clean_text = re.sub('\s+', ' ', text)
    return clean_text.strip()


# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [w for w in tokens if w.isalpha() and w not in stopwords.words('english')]
    return ' '.join(filtered_tokens)


# Function to calculate similarity between two pieces of text
def calculate_similarity(text1, text2):
    cv = CountVectorizer()
    count_matrix = cv.fit_transform([text1, text2])
    match_percentage = cosine_similarity(count_matrix)[0][1] * 100
    return round(match_percentage, 2)

def calculate_and_sort_resumes(zip_file, selected_job_file):
    resumes_with_scores = []  # This will store tuples of (file_name, match_percentage, file_content)

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        jd_text = preprocess_text(extract_and_clean_text(selected_job_file))

        for resume_path in Path(temp_dir).glob('*.pdf'):
            resume_text = preprocess_text(extract_and_clean_text(str(resume_path)))
            match_percentage = calculate_similarity(jd_text, resume_text)

            # Store the file content along with the match score
            with open(resume_path, "rb") as f:
                file_content = f.read()
            resumes_with_scores.append((resume_path.name, match_percentage, base64.b64encode(file_content).decode()))

    # Sort by match percentage in descending order
    sorted_resumes = sorted(resumes_with_scores, key=lambda x: x[1], reverse=True)
    return sorted_resumes

# UI component for ZIP file upload
zip_file = st.file_uploader("Upload a ZIP file containing resumes (PDFs)", type=["zip"])

def gauge(gVal, gTitle="", gMode='gauge+number', gSize="FULL", gTheme="Black",
          grLow=.29, grMid=.69, gcLow='#FF1708', gcMid='#FF9400',
          gcHigh='#1B8720', xpLeft=0, xpRight=1, ypBot=0, ypTop=1,
          arBot=None, arTop=1, pTheme="streamlit", cWidth=True, sFix=None):

    if sFix == "%":
        gaugeVal = round((gVal * 100), 1)
        top_axis_range = (arTop * 100)
        bottom_axis_range = arBot
        low_gauge_range = (grLow * 100)
        mid_gauge_range = (grMid * 100)
    else:
        gaugeVal = gVal
        top_axis_range = arTop
        bottom_axis_range = arBot
        low_gauge_range = grLow
        mid_gauge_range = grMid

    if gaugeVal <= low_gauge_range:
        gaugeColor = gcLow
    elif gaugeVal >= low_gauge_range and gaugeVal <= mid_gauge_range:
        gaugeColor = gcMid
    else:
        gaugeColor = gcHigh

    fig = go.Figure(go.Indicator(
        mode=gMode,
        value=gaugeVal,
        domain={'x': [xpLeft, xpRight], 'y': [ypBot, ypTop]},
        number={"suffix": sFix},
        title={'text': gTitle},
        gauge={
            'axis': {'range': [bottom_axis_range, top_axis_range]},
            'bar': {'color': gaugeColor},
            'steps': [
                {'range': [bottom_axis_range, low_gauge_range], 'color': gcLow},
                {'range': [low_gauge_range, mid_gauge_range], 'color': gcMid},
                {'range': [mid_gauge_range, top_axis_range], 'color': gcHigh},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': gaugeVal
            }
        }
    ))

    config = {'displayModeBar': False}
    fig.update_layout(margin_b=5)
    fig.update_layout(margin_l=20)
    fig.update_layout(margin_r=25)
    fig.update_layout(margin_t=50)

    st.plotly_chart(fig, use_container_width=True)

def display_and_navigate_sorted_resumes(sorted_resumes):
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    current_index = st.session_state.current_index
    total_resumes = len(sorted_resumes)

    resume_name, match_percentage, resume_content_base64 = sorted_resumes[current_index]

    # Display the resume PDF from Base64 content
    pdf_display = f'<iframe src="data:application/pdf;base64,{resume_content_base64}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

    # Navigation
    col1, col2, col3, col4 = st.columns([0.5, 1.5, 2, 1])
    if current_index > 0:  # Only show Previous if not the first resume
        if col2.button("Previous", key="prev"):
            st.session_state.current_index -= 1
    with col3:
        st.write(f"Resume {current_index + 1} of {total_resumes}")
    if current_index < total_resumes - 1:  # Only show Next if not the last resume
        if col4.button("Next", key="next"):
            st.session_state.current_index += 1

    # Display the similarity score as a circular progress bar
    gauge(match_percentage / 100, gTitle="Similarity Score", gMode='gauge+number',
          gcLow='red', gcMid='yellow', gcHigh='green',
          grLow=50, grMid=75, arTop=1, gTheme='Light', sFix="%")

# Adjust the main logic for upload and process
if zip_file is not None:
    sorted_resumes = calculate_and_sort_resumes(zip_file, selected_job_file)
    display_and_navigate_sorted_resumes(sorted_resumes)
else:
    st.write("Please upload a ZIP file containing PDF resumes to proceed.")
