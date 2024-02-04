import streamlit as st
from PIL import Image
from main import ImageProcessor, PDFProcessor  # Replace 'your_module' with the actual module where your ImageProcessor and PDFProcessor classes are defined
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()


genai.configure(api_key=os.getenv("API_KEY"))
llm = genai.GenerativeModel("gemini-pro")


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'application_default_credentials.json'

# title
st.markdown(
    """
    <div style="text-align:center">
        <h1>Q&A with Images & PDF 🤖</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# File upload for image or PDF
file_type = st.radio("Select File Type:", ["Image", "PDF"])
uploaded_file = st.file_uploader("Upload file:", type=["jpg", "jpeg", "png", "pdf"])

# Query text box
query = st.text_input("Enter your question:")

# Display box for generated answer
answer_display = st.empty()

# Creating instances outside the conditional check
image_processor = ImageProcessor(uploaded_file) if uploaded_file and file_type == "Image" else None
pdf_processor = PDFProcessor(uploaded_file) if uploaded_file and file_type == "PDF" else None


if st.button("Generate answer"):
    try:
        if not query:
            raise ValueError("Please provide the question. DO NOT KEEP IT EMPTY!")

        if uploaded_file is not None:
            if file_type == "Image":
                # Image processing
                
                captions = image_processor.get_caption(uploaded_file)
                detections = image_processor.detect_objects(uploaded_file)
                prompt = image_processor.make_prompt(query, captions, detections)
                answer = image_processor.generate_answer(prompt)
                answer_display.markdown(answer)

            elif file_type == "PDF":
                # PDF processing
            
                pdf_vector_stores = pdf_processor.create_embedding_df(uploaded_file)
                relevant_passage = pdf_processor.find_best_passage(query, pdf_vector_stores)
                prompt = pdf_processor.make_prompt(query, relevant_passage)
                answer = pdf_processor.generate_answer(prompt)
                answer_display.markdown(answer)


    except ValueError as e:
        st.warning(str(e))