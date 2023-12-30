import streamlit as st
import pandas as pd
import zipfile
import io
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import BSHTMLLoader, JSONLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter,CharacterTextSplitter
import json
from pathlib import Path
from langchain.embeddings import HuggingFaceInstructEmbeddings,GPT4AllEmbeddings,HuggingFaceEmbeddings,OllamaEmbeddings,SpacyEmbeddings,TensorflowHubEmbeddings
import tempfile
from langchain.vectorstores import Chroma,FAISS,LanceDB

def process_file(file_stream, file_name, progress_bar):
    file_extension = os.path.splitext(file_name)[1].lower()
    text_chunks = []  # List to store text chunks

    # Start progress bar at 50% for processing
    progress_bar.progress(0.5)

    # Write content to a temporary file for loaders that require a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(file_stream.read())
        tmp_file_path = tmp_file.name

    try:
        if file_extension in [".csv", ".xls", ".xlsx"]:
            # For CSV and Excel, assume we are splitting by rows (as an example)
            if file_extension == ".csv":
                df = pd.read_csv(tmp_file_path)
            else:  # Excel files
                df = pd.read_excel(tmp_file_path)

            # Example: Splitting every 10 rows
            for start in range(0, len(df), 10):
                chunks = df[start:start + 10]
                text_chunks.extend(chunks)
                # st.write(chunks, max_rows=2)
            return text_chunks
                

        elif file_extension == ".pdf":
            # For PDFs, use PyPDFLoader and then further split text
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load_and_split()  # This returns a list of Document objects
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

            for document in documents:
                if document.page_content:  # Ensure there is text content in the document
                    text_content = document.page_content  # Extract text content from Document
                    chunks = text_splitter.split_text(text_content)  # Split the text content
                    text_chunks.extend(chunks)
                    # for split in chunks:
                    #     st.text(split)  # Display each split
            return text_chunks

        elif file_extension in [".html", ".htm"]:
            # Handle HTML files using Langchain's HTMLHeaderTextSplitter
            from langchain.text_splitter import HTMLHeaderTextSplitter
            with open(tmp_file_path, 'r') as file:
                html_content = file.read()
            headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]
            html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            chunks = html_splitter.split_text(html_content)
            text_chunks.extend(chunks)
            # st.write(chunks)
            return text_chunks

        elif file_extension == ".txt":
            # Handle plain text files using CharacterTextSplitter
            from langchain.text_splitter import CharacterTextSplitter
            with open(tmp_file_path, 'r') as file:
                text_content = file.read()
            text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.create_documents([text_content])
            text_chunks.extend(chunks)
            # st.write(chunks)
            return text_chunks

        # elif file_extension in ["py", "java", "js", "cpp"]:  # Add other code file extensions as needed
        #     # Handle code files using CodeTextSplitter
        #     from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
        #     with open(tmp_file_path, 'r') as file:
        #         code_content = file.read()
        #     language_map = {"py": Language.PYTHON, "java": Language.JAVA, "js": Language.JS, "cpp": Language.CPP}
        #     language = language_map.get(file_extension, Language.PYTHON)  # Default to Python if not mapped
        #     code_splitter = RecursiveCharacterTextSplitter.from_language(language=language, chunk_size=50, chunk_overlap=0)
        #     code_splits = code_splitter.create_documents([code_content])
        #     st.write(code_splits)
        else:
            return "Unsupported file format!"


    finally:
        # Complete the progress bar at 100%
        progress_bar.progress(1.0)
        # Clean up the temporary file
        os.remove(tmp_file_path)


# def process_embedding(embedding_type, text_chunks):

#     try:
#         embeddings = assign_embedding(embedding_type)
#         if embeddings and text_chunks:
#             # Extract text content from Document objects if necessary
#             texts = [chunk.text if hasattr(chunk, 'text') else str(chunk) for chunk in text_chunks]
#             query_result = embeddings.embed_documents(texts)
#             st.write(query_result)
#         else:
#             st.write("No text data available or failed to load embeddings.")
#     except Exception as e:
#         st.error(f"Error with {embedding_type} embeddings: {e}")


def assign_embedding(embedding_type):
    if embedding_type == 'huggingface':
        return HuggingFaceEmbeddings()
    elif embedding_type == 'gpt4all':
        return GPT4AllEmbeddings()
    elif embedding_type == 'HuggingFaceInstruct': #https://huggingface.co/spaces/mteb/leaderboard # pip install --upgrade transformers accelerate
        return HuggingFaceInstructEmbeddings(model_name="intfloat/e5-mistral-7b-instruct",model_kwargs={"device":"cpu"})
    elif embedding_type == 'ollama':
        return OllamaEmbeddings()
    elif embedding_type == 'spacy':
        return SpacyEmbeddings()
    elif embedding_type == 'tensorflow_hub':
        return TensorflowHubEmbeddings()
    else:
        return None



def main():
    # Initialize session state
    if 'text' not in st.session_state:
        st.session_state['text'] = None
    if 'embeddings' not in st.session_state:
        st.session_state['embeddings'] = None

    # Configure page layout
    st.set_page_config(layout="wide")

    # Title of the app
    st.title('File Upload and Display App')

    # Instruction
    st.write('Upload a Text, PDF, Word, Excel, CSV, or ZIP File to View Its Contents')

    # File uploader widget
    st.header('Single File Upload')
    uploaded_file = st.file_uploader('Upload a file')

    if uploaded_file is not None:
        # Initialize progress bar
        progress_bar = st.progress(0)

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        # Check if the uploaded file is a ZIP file
        if file_extension == ".zip":
                all_text_chunks = []  # List to store text chunks from all files in the ZIP
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall("temp_dir")
                    for idx, filename in enumerate(zip_ref.namelist()):
                        file_path = os.path.join("temp_dir", filename)
                        if os.path.isfile(file_path):
                            st.write(f"Processing {filename}:")
                            with open(file_path, "rb") as f:
                                progress_bar.progress(0.5 * (idx + 1) / len(zip_ref.namelist()))
                                text = process_file(io.BytesIO(f.read()), filename, progress_bar)
                                all_text_chunks.extend(text)
                                st.write(all_text_chunks)
                            os.remove(file_path)
                    os.rmdir("temp_dir")
        else:
             text = process_file(uploaded_file, uploaded_file.name, progress_bar)
             st.write(text)
    
    st.header('Choose an Embedding Model')
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        if st.button('HuggingFace Embeddings'):
            process_embedding('huggingface', text)

    with col2:
        if st.button('GPT4All Embeddings'):
           process_embedding('gpt4all', text)

    with col3:
        if st.button('Ollama Embeddings'):
            process_embedding('ollama', text)

    with col4:
        if st.button('Spacy Embeddings'):
           process_embedding('spacy', text)

    with col5:
        if st.button('Tensorflow Hub Embeddings'):
           process_embedding('tensorflow_hub', text)
           
    with col6:
        if st.button('HuggingFaceInstruct Embeddings'):
           process_embedding('HuggingFaceInstruct', text)


    st.header('Choose a DB')
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button('FAISS'):
            pass
           
    with col2:
        if st.button('Chroma'):
           pass

    with col3:
        if st.button('LanceDB'):
            pass



if __name__ == "__main__":
    main()