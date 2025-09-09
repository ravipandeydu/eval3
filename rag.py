import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredImageLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Global variables for RAG components
vector_store = None
rag_chain = None
embeddings = None
llm = None
prompt = None

# Initialize the embeddings model and LLM
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define the prompt template
prompt_template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Be concise.

Context: {context} 
Question: {question} 

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create the RAG chain
rag_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# This helper function formats the retrieved documents into a single string.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_documents_from_folder(docs_folder="docs"):
    """Load all PDF documents from the specified folder."""
    all_docs = []

    if os.path.exists(docs_folder):
        # Handle pdf files
        pdf_files = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]
        print(f"Found {len(pdf_files)} PDF files in {docs_folder} folder: {pdf_files}")

        # Load each PDF file
        for pdf_file in pdf_files:
            pdf_path = os.path.join(docs_folder, pdf_file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} pages from {pdf_file}")

        # handle docx files
        docx_files = [f for f in os.listdir(docs_folder) if f.endswith(".docx")]
        print(
            f"Found {len(docx_files)} DOCX files in {docs_folder} folder: {docx_files}"
        )

        # Load each DOCX file
        for docx_file in docx_files:
            docx_path = os.path.join(docs_folder, docx_file)
            loader = Docx2txtLoader(docx_path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} pages from {docx_file}")

        # handle txt files
        txt_files = [f for f in os.listdir(docs_folder) if f.endswith(".txt")]
        print(f"Found {len(txt_files)} TXT files in {docs_folder} folder: {txt_files}")

        # Load each TXT file
        for txt_file in txt_files:
            txt_path = os.path.join(docs_folder, txt_file)
            loader = TextLoader(txt_path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} pages from {txt_file}")

        # handle html files
        html_files = [f for f in os.listdir(docs_folder) if f.endswith(".html")]
        print(
            f"Found {len(html_files)} HTML files in {docs_folder} folder: {html_files}"
        )
        # Load each HTML file
        for html_file in html_files:
            html_path = os.path.join(docs_folder, html_file)
            loader = UnstructuredHTMLLoader(html_path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} pages from {html_file}")

        # handle Image files
        image_files = [
            f
            for f in os.listdir(docs_folder)
            if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")
        ]
        print(
            f"Found {len(image_files)} Image files in {docs_folder} folder: {image_files}"
        )

        for image_file in image_files:
            image_path = os.path.join(docs_folder, image_file)
            loader = UnstructuredImageLoader(image_path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} pages from {image_file}")

        print(f"Total loaded {len(all_docs)} pages from all documents.")
    else:
        print(f"Error: {docs_folder} folder not found!")

    return all_docs


def initialize_rag_system():
    """Initialize or reinitialize the RAG system with current documents."""
    global vector_store, rag_chain

    # Load all documents
    all_docs = load_documents_from_folder()

    if not all_docs:
        print("No documents found. RAG system not initialized.")
        return False

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"Split all documents into {len(splits)} chunks.")

    # Create the vector store from the document splits and embeddings
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
    print("Vector store created successfully.")

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Define the RAG chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return True


# Initialize the RAG system on startup
initialize_rag_system()


# Function to get answer from RAG chain
def get_rag_answer(question: str) -> str:
    """Get answer from the RAG chain for a given question."""
    global rag_chain

    if rag_chain is None:
        return "RAG system not initialized. Please upload some documents first."

    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        return f"Error processing question: {str(e)}"


def reload_rag_system():
    """Reload the RAG system to include newly uploaded documents."""
    print("Reloading RAG system with updated documents...")
    success = initialize_rag_system()
    if success:
        print("RAG system reloaded successfully.")
        return True
    else:
        print("Failed to reload RAG system.")
        return False
