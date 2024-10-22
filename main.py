from prefect import flow, task
from typing import List, Optional
from pathlib import Path
import boto3
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Weaviate
import weaviate
from typing import List, Dict
from langchain.schema import Document

# Initialize the S3 client
def init_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name='us-west-1'
    )

# # Task to extract text from a PDF file
# def extract_text_from_pdf(pdf_stream: BytesIO) -> str:
#     reader = PyPDF2.PdfReader(pdf_stream)
#     all_text = ""
#     for page in reader.pages:
#         all_text += page.extract_text()
#     return all_text

# # Task to extract text from a DOC file
# def extract_text_from_doc(doc_stream: BytesIO) -> str:
#     all_text = ""
#     doc = Document(doc_stream)
#     for para in doc.paragraphs:
#         all_text += para.text + "\n"
#     return all_text

# # Task to extract text from a TXT file
# def extract_text_from_txt(txt_stream: BytesIO) -> str:
#     return txt_stream.read().decode('utf-8')

# # Task to determine file type and extract text
# @task
# def extract_text_from_s3(object_name:str) -> str:
#     s3_client = init_s3_client()
#     bucket_name="wachatbot-aiplanet"
#     # Download the object from S3
#     response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
    
#     # Read the file content as a binary stream
#     file_stream = BytesIO(response['Body'].read())
#     file_ext = object_name.split('.')[-1].lower()  # Get the file extension
    
#     if file_ext == 'pdf':
#         # Extract text from a PDF
#         return extract_text_from_pdf(file_stream)
#     elif file_ext == 'docx':
#         # Extract text from a DOCX file
#         return extract_text_from_doc(file_stream)
#     elif file_ext == 'txt':
#         # Extract text from a TXT file
#         return extract_text_from_txt(file_stream)
#     else:
#         raise ValueError(f"Unsupported file extension: {file_ext}")

@task
def extract_text_from_pdf(file_path: str) -> str:
    endpoint = "https://fullstackai.cognitiveservices.azure.com/"
    key = "d38bcbcda9484b1c9204373fda81d060"
    
    # Initialize the Document Analysis client
    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    # Read the file content
    with open(file_path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document("prebuilt-document", document=f)
    
    # Get the result of the analysis
    result = poller.result()

    # Extract and concatenate the text from all pages
    extracted_text = ""
    for page in result.pages:
        for line in page.lines:
            extracted_text += line.content + "\n"

    return extracted_text

@task 
def download_pdf_from_s3(object_name: str):
   
    s3_client = init_s3_client()

    bucket_name="wachatbot-aiplanet"
    local_folder="downloaded_documents"
    # Create the folder if it doesn't exist
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
        print(f"Folder '{local_folder}' created.")
    
    # Create the full local file path
    local_file_path = os.path.join(local_folder, os.path.basename(object_name))
    
    # Download the file from S3 and save it locally
    with open(local_file_path, 'wb') as f:
        s3_client.download_fileobj(bucket_name, object_name, f)
    print(f"File downloaded locally as {local_file_path}")
    
    return local_file_path


@task
def chunk_documents(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    return documents
   
@task
def store_documents_in_weaviate(
    chunked_docs: List[Document],
) -> None:
    retriever_config = {
        "type": "Weaviate",
        "kwargs": {
            "weaviate_url": "https://weaviate.aimarketplace.co",
            "weaviate_api_key": "5d92168b96ea4b9bb553b02cd14157b7",
            "index_name": "Id_knowledge_base_testing_c0f13c"
        }
    }

    embedding_config = {
        "type": "HuggingFaceInferenceAPIEmbeddings",
        "kwargs": {
            "api_key": "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",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        }
    }
    # Create embedding model
    embedding_model = HuggingFaceInferenceAPIEmbeddings(**embedding_config['kwargs'])
    print("Embedding model initialized.")

    # Initialize Weaviate client
    client = weaviate.Client(
        url=retriever_config['kwargs']['weaviate_url'],
        auth_client_secret=weaviate.AuthApiKey(api_key=retriever_config['kwargs']['weaviate_api_key'])
    )
    print("Weaviate client initialized.")

    # # Store in Weaviate
    Weaviate.from_documents(
        documents=chunked_docs,
        embedding=embedding_model,
        client=client,
        index_name=retriever_config['kwargs']['index_name']
    )
    print(f"Chunks stored in Weaviate index: {retriever_config['kwargs']['index_name']}")

@flow
def execute_data_pipeline(
    object_name: Optional[str] = None,
):
    file_path = download_pdf_from_s3(object_name)
    text = extract_text_from_pdf(file_path)
    chunk = chunk_documents(text)
    store_documents_in_weaviate(chunk)

    print(f"Extracted chunk: {chunk}")
    
if __name__ == "__main__":
    object_name="knowledge_base_test/scansmpl.pdf"
    execute_data_pipeline(object_name=object_name)