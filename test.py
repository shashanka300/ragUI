import streamlit as st
import pandas as pd
import zipfile
import io
import os
import ast
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import BSHTMLLoader, JSONLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter,CharacterTextSplitter
import json
from pathlib import Path
from langchain.embeddings import HuggingFaceInstructEmbeddings,GPT4AllEmbeddings,HuggingFaceEmbeddings,OllamaEmbeddings,SpacyEmbeddings,TensorflowHubEmbeddings
import tempfile
from langchain.vectorstores import Chroma,FAISS,LanceDB
from IBM import IBM_Bam
from retrival import rag_fussion_compression
from evaluation import metrics

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
            # print(text_chunks)
            return text_chunks

        else:
            return "Unsupported file format!"


    finally:
        # Complete the progress bar at 100%
        progress_bar.progress(1.0)
        # Clean up the temporary file
        os.remove(tmp_file_path)


def process_embedding(embedding_type, text_chunks,multi_line_text,vstore):

    try:
        embeddings = assign_embedding(embedding_type)
        if embeddings and text_chunks:
            db = vstore.from_documents(text_chunks, embeddings)
            retriever = db.as_retriever()
            docs = retriever.get_relevant_documents(multi_line_text)
            st.write(docs[0].page_content)
            return retriever
        else:
            st.write("No text data available or failed to load embeddings.")
    except Exception as e:
        st.error(f"Error with {embedding_type} embeddings: {e}")



        
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

    # Initialize session state for embedding type
    if 'embeddings_type' not in st.session_state:
        st.session_state['embeddings_type'] = None
    
    if 'retriever' not in st.session_state:
        st.session_state['retriever'] = None
    
    if 'result' not in st.session_state:
        st.session_state['result'] = None

    # output = 'please enter a prompt'
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
           st.session_state['embeddings_type'] = 'huggingface'

    with col2:
        if st.button('GPT4All Embeddings'):
           st.session_state['embeddings_type'] = 'gpt4all'

    with col3:
        if st.button('Ollama Embeddings'):
            st.session_state['embeddings_type'] = 'ollama'

    with col4:
        if st.button('Spacy Embeddings'):
           st.session_state['embeddings_type'] = 'spacy'

    with col5:
        if st.button('Tensorflow Hub Embeddings'):
           st.session_state['embeddings_type'] = 'tensorflow_hub'
           
    with col6:
        if st.button('HuggingFaceInstruct Embeddings'):
           st.session_state['embeddings_type'] = 'HuggingFaceInstruct'

    # st.text(embeddings)

    # Multi-line text area
    multi_line_text = st.text_area("Enter your text here (multi-line):")

    st.header('Choose a DB')
    col1, col2 = st.columns(2)

    with col1:
        if st.button('FAISS'):
            if 'embeddings_type' in st.session_state:
                retriever = process_embedding(st.session_state['embeddings_type'], text, multi_line_text, FAISS)
                st.session_state['retriever'] = retriever

            else:
                st.error("Please select an embedding model first.")
                      
    with col2:
        if st.button('Chroma'):
            if 'embeddings_type' in st.session_state:
                retriever = process_embedding(st.session_state['embeddings_type'], text, multi_line_text, Chroma)
                st.session_state['retriever'] = retriever
            else:
                st.error("Please select an embedding model first.")
           
    # Multi-line text area 
    prompt = st.text_area("Enter your prompt for Rag (multi-line):")
    
    st.header("Single Digit Input")

    # Add a text input widget for the user to enter a single digit
    count = st.text_input("Enter a single digit number between 2-9:")

    # Validate the input to ensure it's a single digit integer
    if count.isdigit() and len(count) == 1:
        st.write("You entered:", int(count))  # Convert to integer for further use
    else:
        st.error("Please enter a single digit number.")
    
    st.header('Choose a Retrival type')
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button('Rag_fussion'):
            f_prompt = rag_fussion_compression.query_prompt(prompt, count='3')
            result = IBM_Bam.U_bam(f_prompt)
            output = result.replace('"rephrased_questions":',"")
            multi_query = ast.literal_eval(output)
            multi_query.insert(0,prompt)
            st.write(multi_query)

            mulri_results=rag_fussion_compression.fusion_formator(st.session_state['retriever'],multi_query)

            ranked_result = rag_fussion_compression.reciprocal_rank_fusion(mulri_results)

            txt = []
            for data in ranked_result[:3]:
                txt.append(data[0]['text'])

            result_P = rag_fussion_compression.result_prompt(prompt, txt)
            result =  IBM_Bam.U_bam(result_P)
            st.session_state['result'] = result
            st.write(result)

    with col2:
        if st.button('Rag_fussion_weighted'):
            f_prompt = rag_fussion_compression.query_prompt(prompt, count='3')
            result = IBM_Bam.U_bam(f_prompt)
            output = result.replace('"rephrased_questions":',"")
            multi_query = ast.literal_eval(output)
            multi_query.insert(0,prompt)
            st.write(multi_query)

            mulri_results=rag_fussion_compression.fusion_formator(st.session_state['retriever'],multi_query)

            ranked_result = rag_fussion_compression.reciprocal_rank_fusion_weighted(mulri_results)

            txt = []
            for data in ranked_result[:3]:
                txt.append(data[0]['text'])

            result_P = rag_fussion_compression.result_prompt(prompt, txt)
            result =  IBM_Bam.U_bam(result_P)
            st.session_state['result'] = result
            st.write(result)

    with col3:
        if st.button('Rag_fussion_Contextual_compression'):
            f_prompt = rag_fussion_compression.query_prompt(prompt, count='3')
            result = IBM_Bam.U_bam(f_prompt)
            output = result.replace('"rephrased_questions":',"")
            multi_query = ast.literal_eval(output)
            multi_query.insert(0,prompt)
            st.write(multi_query)

            mulri_results=rag_fussion_compression.fusion_formator_compressed(IBM_Bam.langchain_model,st.session_state['retriever'],multi_query)

            ranked_result = rag_fussion_compression.reciprocal_rank_fusion_weighted(mulri_results)

            txt = []
            for data in ranked_result[:3]:
                txt.append(data[0]['text'])

            result_P = rag_fussion_compression.result_prompt(prompt, txt)
            result =  IBM_Bam.U_bam(result_P)
            st.session_state['result'] = result
            st.write(result)
    

       # Multi-line text area 
    references = st.text_area(f"the ground truth answer for you question - {prompt}")

    st.header('Choose an evaluation Metric')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button('Blue'):
           blue = metrics.bleu_score(references,st.session_state['result'])
           st.write(blue)

    with col2:
        if st.button('Rouge 1 2 L'):
           Rouge_1,Rouge_2,Rouge_L = metrics.rouge_score(references,st.session_state['result'])
           st.write(Rouge_1,Rouge_2,Rouge_L)

    with col3:
        if st.button('Precission Recall'):
            precision, recall = metrics.precision_recall(references,st.session_state['result'])
            st.write(precision, recall)
    
    with col4:
        st.write('Dosent need ground truth')
        if st.button('Flesch-Kincaid reading'):
            blue = metrics.readability(st.session_state['result'])
            st.write(blue)

if __name__ == "__main__":
    main()
