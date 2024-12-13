import os
import pandas as pd
import json
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredImageLoader,
    UnstructuredFileLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)

# from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_groq import ChatGroq

app = Flask(__name__)

# Use absolute path for FAISS
DB_PATH = os.path.join(os.getcwd(), "FAISS_1")
UPLOAD_FOLDER = "documents"

class DocumentProcessor:
    """Class to process documents and extract text."""
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def list_files_recursively(self):
        """List all file paths recursively within a directory."""
        file_paths = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                file_paths.append(os.path.join(subdir, file))
        return file_paths

    def process_files(self, file_paths):
        """Process files and extract text based on file type."""
        text = []
        loaders = {
            ".pdf": PyPDFLoader,
            ".csv": CSVLoader,
            ".docx": Docx2txtLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xlsm": UnstructuredExcelLoader,
            ".jpg": UnstructuredImageLoader,
            ".txt": UnstructuredFileLoader,
        }

        for path in file_paths:
            file_extension = os.path.splitext(path)[1].lower()
            if file_extension in loaders:
                loader = loaders[file_extension](path) if file_extension not in [".xlsx", ".xlsm"] else loaders[file_extension](path, mode="elements")
                file_data = loader.load()
                filtered_data = filter_complex_metadata(file_data)
                text.extend(filtered_data)
            else:
                print(f"Unsupported file format: {path}")  
        return text

class DocumentSplitter:
    """Class to split documents into smaller chunks."""
    def __init__(self, chunk_size=1000, chunk_overlap=20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_docs(self, documents):
        """Split documents into smaller chunks for processing."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return text_splitter.split_documents(documents)

def initialize_vector_store(upload_folder):
    """Initialize vector store from documents in the upload folder."""
    # Ensure upload folder exists
    os.makedirs(upload_folder, exist_ok=True)

    # Document processing
    processor = DocumentProcessor(upload_folder)
    file_paths = processor.list_files_recursively()

    if file_paths:
        print(f"Found {len(file_paths)} files to process")
        text = processor.process_files(file_paths)
        splitter = DocumentSplitter()
        doc = splitter.split_docs(text)

        # Create a vector store
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create or update FAISS vector store
        vectorstore = FAISS.from_documents(doc, embedding_function)
        vectorstore.save_local(DB_PATH)

        return vectorstore
    print("No files found in the upload folder")
    return None

def create_retrieval_chain():
    """Create a retrieval chain for document querying."""
    # Initialize embeddings 
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Check if FAISS index exists
    faiss_index_path = os.path.join(DB_PATH, "index.faiss")
    if not os.path.exists(faiss_index_path):
        print("No existing FAISS index. Attempting to create one...")
        initialize_vector_store(UPLOAD_FOLDER)
        if not os.path.exists(faiss_index_path):
            return None

    try:
        # Load existing vector store
        vectorstore = FAISS.load_local(DB_PATH, embeddings=embedding_function, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()

        # Initialize LLM
        llm = ChatGroq(model="Gemma2-9b-It", api_key="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")

        # Templates as strings
        prompt_template = PromptTemplate(
        input_variables=["query","context"],
        template="""
        You are a professional document generator tasked with creating a comprehensive document structure.

        Instructions:
        1. Maintain the following static headings and subheadings:
        1. Executive Summary
        2. Company Background
        2.1 About our company
        2.2 Our Journey
        2.3 Facts & Figures

        2. Generate dynamic content for the under the appropriate sections.
        3. Add up to 4 additional relevant headings and subheadings that are contextually appropriate to the.
        4. Ensure the final document includes these static headings:
        6. Why our company?
        7. Team Structure
        8. Phase 1-Estimation
        9. Pricing and Estimation
        10. Relevant Experience and Case Studies

        Topic Details:
        Based on the topic develop a comprehensive, well-structured document that provides:
        - In-depth insights
        - Relevant market research
        - Strategic analysis
        - Potential opportunities and challenges

        Formatting Guidelines:
        - Use clear, professional language
        - Provide specific, actionable insights
        - Ensure logical flow between sections
        - Include data-driven observations where possible
        context:
        {context}
        query:{query}
        Generate a document that comprehensively addresses the while maintaining the specified structure."""
        
    )

        # Create retrieval chain
        chain = (
            {"context": retriever, "query": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        return chain
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

@app.route('/query', methods=['POST'])
def query_documents():
    """Endpoint to query uploaded documents."""
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    chain = create_retrieval_chain()
    if not chain:
        print("Retrieval chain could not be created. Check FAISS index.")
        return jsonify({"error": "No documents indexed. Please ensure documents are in the upload folder."}), 400
    
    try:
        output = chain.invoke(query)
        print(f"Query output: {output}")
        return jsonify({"response": output})
    except Exception as e:
        print(f"Error during query processing: {e}")
        return jsonify({"error": str(e)}), 500
    
initialize_vector_store(UPLOAD_FOLDER)

if __name__ == '__main__':
    # Automatically initialize vector store on startu
    app.run(debug=True)

