import textwrap
import numpy as np
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PyPDF2 import PdfReader
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image
import torch
from IPython.display import Markdown
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


# Used to securely store your API key
# from google.colab import userdata





class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path

    def get_caption(self, image_path):
        # Implement image captioning logic here
        """
        Generates a short caption for the provided image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: A string representing the caption for the image.
        """
        image = Image.open(image_path).convert('RGB')

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"  # cuda

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption



    def detect_objects(self, image_path):
        # Implement object detection logic here
        """
        Detects objects in the provided image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: A string with all the detected objects. Each object as '[x1, x2, y1, y2, class_name, confindence_score]'.
        """
        image = Image.open(image_path).convert('RGB')

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        return detections



    def make_prompt(self, query, image_captions, objects_detections):
        # Implement prompt creation logic here
        escaped_captions = image_captions.replace("'", "").replace('"', "").replace("\n", " ")
        escaped_objects = objects_detections.replace("'", "").replace('"', "").replace("\n", " ")
        prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the image captions and objects detected included below. \
        Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
        strike a friendly and conversational tone. \
        If the image captions or objects detected are irrelevant to the answer, you may ignore them.
        QUESTION: '{query}'
        IMAGE CAPTIONS: '{image_captions}'
        OBJECTS DETECTED: '{objects_detected}'

            ANSWER:
        """).format(query=query, image_captions=escaped_captions, objects_detected=escaped_objects)

        return prompt



    def generate_answer(self, prompt):
        # Implement answer generation logic here
        model = genai.GenerativeModel('gemini-pro')
        answer = model.generate_content(prompt)
        
        return answer.text

        




class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def create_embedding_df(self, pdf_path):
        # Implement PDF content vector store creation logic here
        # Provide the path of the PDF file
        pdfreader = PdfReader(pdf_path)

        # Read text from PDF and divide it into smaller chunks
        documents = []
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                # Create a document for each page
                document = {
                    "Title": f"Page {i+1}",  # Use the page number as the title
                    "Text": content
                }
                documents.append(document)

        # Create a DataFrame from the documents
        df = pd.DataFrame(documents)

        # Define the model
        model = 'models/embedding-001'

        # Define a function to generate embeddings
        def embed_fn(title, text):
            return genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document",
                title=title
            )["embedding"]

        # Generate embeddings for each document and store them in the DataFrame
        df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)

        return df



    def find_best_passage(self, query, dataframe):
        # Implement logic to find the best passage based on query
        """
        Compute the distances between the query and each document in the dataframe
        using the dot product.
        """
        model = 'models/embedding-001'
        query_embedding = genai.embed_content(model=model,
                                              content=query,
                                              task_type="retrieval_query")
        dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
        idx = np.argmax(dot_products)
        # Return text from index with max value
        return dataframe.iloc[idx]['Text'] 



    def make_prompt(self, query, relevant_passage):
        # Implement prompt creation logic for PDF processing
          escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
          prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
          Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
          However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
          strike a friendly and converstional tone. \
          If the passage is irrelevant to the answer, you may ignore it.
          QUESTION: '{query}'
          PASSAGE: '{relevant_passage}'

            ANSWER:
          """).format(query=query, relevant_passage=escaped)

          return prompt



    def generate_answer(self, prompt):
        # Implement answer generation logic for PDF processing
        model = genai.GenerativeModel('gemini-pro')
        answer = model.generate_content(prompt)

        return answer.text






