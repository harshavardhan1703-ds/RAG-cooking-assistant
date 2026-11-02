import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load secret API keys from the local .env file
load_dotenv()
print("Environment variables loaded successfully.")


# STEP 1: FUNCTION TO READ ALL PDF FILES
def process_all_pdfs(pdf_directory):
    """
    This function looks inside the specified folder, 
    finds all PDF files, and extracts text page by page.
    """
    print(f"Starting to scan directory for PDFs: {pdf_directory}")
    
    all_extracted_documents = []
    target_directory = Path(pdf_directory)
    
    # Get a list of all PDF files in the folder using pathlib
    found_pdf_files = list(target_directory.glob("*.pdf"))
    print(f"Found {len(found_pdf_files)} PDF file(s) in the folder.")
    
    # Loop through each PDF file found
    for single_pdf_file in found_pdf_files:
        print(f"Reading file: {single_pdf_file.name}")
        
        # Initialize the PDF loader using PyMuPDFLoader for reliable text extraction
        pdf_loader = PyMuPDFLoader(str(single_pdf_file))
        
        # Load all pages from the current PDF
        loaded_pages = pdf_loader.load()
        print(f"Successfully loaded {len(loaded_pages)} pages from {single_pdf_file.name}")
        
        # Add these pages to our master document list
        all_extracted_documents.extend(loaded_pages)
        
    print(f"Total pages loaded across all PDFs: {len(all_extracted_documents)}")
    return all_extracted_documents


# STEP 2: FUNCTION TO SPLIT TEXT INTO CHUNKS
def split_documents(raw_documents):
    """
    Takes raw document pages and breaks them down into 
    smaller, manageable text chunks for the vector database.
    """
    print("Initializing text splitter...")
    
    text_splitter_tool = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    document_chunks = text_splitter_tool.split_documents(raw_documents)
    print(f"Split raw documents into {len(document_chunks)} total text chunks.")
    return document_chunks


# STEP 3: EMBEDDING MANAGER CLASS
class EmbeddingManager:
    """
    Responsible for loading the Sentence Transformer model 
    and converting text chunks into numerical vector embeddings.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading sentence transformer embedding model: {model_name}")
        self.model_name = model_name
        self.sentence_model = SentenceTransformer(model_name)
        print("Embedding model loaded and ready.")
        
    def generate_embeddings(self, list_of_texts):
        print(f"Generating vector embeddings for {len(list_of_texts)} text items...")
        vector_embeddings = self.sentence_model.encode(list_of_texts)
        print("Embeddings generated successfully.")
        return vector_embeddings


# STEP 4: VECTOR STORE CLASS (USING CHROMADB)
class VectorStore:
    """
    Manages the ChromaDB persistent database where text chunks 
    and their vector embeddings are stored for fast searching.
    """
    def __init__(self, collection_name="pdf_documents", persist_directory="data/vector_store"):
        print(f"Initializing Vector Store at directory: {persist_directory}")
        
        # Ensure the directory exists on the computer
        os.makedirs(persist_directory, exist_ok=True)
        
        # Create a persistent client for ChromaDB
        self.db_client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create our document collection
        self.collection = self.db_client.get_or_create_collection(name=collection_name)
        
        current_document_count = self.collection.count()
        print(f"Vector store ready. Existing documents in collection: {current_document_count}")

    def add_documents(self, document_chunks_list, computed_embeddings):
        """
        Saves text chunks, their embeddings, and unique IDs into ChromaDB.
        """
        print("Preparing documents and IDs for database insertion...")
        
        # Extract plain text from document chunk objects
        extracted_text_list = [chunk.page_content for chunk in document_chunks_list]
        
        # Generate a unique ID string for every single chunk
        generated_ids = [f"chunk_id_{index}" for index in range(len(document_chunks_list))]
        
        print(f"Adding {len(extracted_text_list)} items into the ChromaDB collection...")
        self.collection.add(
            documents=extracted_text_list,
            embeddings=computed_embeddings.tolist(),
            ids=generated_ids
        )
        print("Documents successfully added to the vector store database.")


# STEP 5: RAG RETRIEVER CLASS
class RAGRetriever:
    """
    Searches the vector database using a user question 
    to find the most relevant cookbook text chunks.
    """
    def __init__(self, vector_store_instance, embedding_manager_instance):
        print("Initializing RAG Retriever...")
        self.vector_store = vector_store_instance
        self.embedding_manager = embedding_manager_instance

    def retreive(self, search_query, top_k=5):
        print(f"Retriever received search query: '{search_query}'")
        
        # Convert the user's question into a vector embedding
        query_vector_embedding = self.embedding_manager.generate_embeddings([search_query])[0]
        
        print(f"Querying ChromaDB for the top {top_k} most relevant matches...")
        database_query_results = self.vector_store.collection.query(
            query_embeddings=[query_vector_embedding.tolist()],
            n_results=top_k
        )
        
        # Format the retrieved results neatly into a list of dictionaries
        formatted_retrieved_documents = []
        
        if database_query_results['documents'] and database_query_results['documents'][0]:
            matching_texts = database_query_results['documents'][0]
            print(f"Found {len(matching_texts)} matching text chunks from database.")
            
            for matching_text_item in matching_texts:
                formatted_retrieved_documents.append({'content': matching_text_item})
        else:
            print("No matching documents found in the database for this query.")
            
        return formatted_retrieved_documents


# STEP 6: SIMPLE RAG FUNCTION FOR LLM ANSWER GENERATION
def rag_simple(user_query, retriever_instance, language_model, top_k=5):
    """
    Gathers relevant context chunks using the retriever, 
    builds a prompt, and asks Groq LLM to generate an answer.
    """
    print(f"Running simple RAG pipeline for question: '{user_query}'")
    
    # Step A: Retrieve relevant context chunks
    retrieved_chunks = retriever_instance.retreive(user_query, top_k=top_k)
    
    # Step B: Combine chunk contents into a single context string
    combined_context_text = "\n\n".join([chunk['content'] for chunk in retrieved_chunks])
    
    if not combined_context_text:
        print("Context is empty. Returning default message.")
        return "No relevant information found."

    # Step C: Construct the final prompt for the language model
    constructed_prompt = f"""Use the context below to answer the question accurately. 
Start your response exactly with "Here's how to cook {user_query}:" followed by the ingredients and step-by-step instructions.

Context:
{combined_context_text}

Question: {user_query}
Answer:"""

    print("Sending constructed prompt to Groq LLM...")
    llm_response = language_model.invoke(constructed_prompt)
    print("Received response from Groq LLM.")
    
    return llm_response.content


# STEP 7: AUTO-INITIALIZATION HELPER FUNCTION
def initialize_vector_store_if_empty(vectorstore_instance, embedding_manager_instance, pdf_directory="data/pdf_documents"):
    """
    Checks if the vector database is empty. If it is, 
    it automatically processes, chunks, embeds, and stores the PDFs.
    """
    print("Checking if vector database needs initialization...")
    existing_count = vectorstore_instance.collection.count()
    
    if existing_count == 0:
        print("Vector store is completely empty! Starting automatic PDF processing...")
        
        raw_pdf_docs = process_all_pdfs(pdf_directory)
        
        if raw_pdf_docs:
            split_text_chunks = split_documents(raw_pdf_docs)
            
            text_contents_only = [chunk.page_content for chunk in split_text_chunks]
            computed_embeddings = embedding_manager_instance.generate_embeddings(text_contents_only)
            
            vectorstore_instance.add_documents(split_text_chunks, computed_embeddings)
            print("Automatic PDF ingestion and embedding completed successfully!")
        else:
            print("Warning: No documents were found to process.")
    else:
        print(f"Vector store already contains {existing_count} chunks. Skipping ingestion.")


# STEP 8: MAIN EXECUTION SCRIPT (CLI TEST MODE)
if __name__ == "__main__":
    print("STARTING RAG PROJECT IN TERMINAL TEST MODE")
    
    # 1. Initialize embedding manager and vector store
    my_embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    my_vector_store = VectorStore(collection_name="pdf_documents", persist_directory="data/vector_store")
    
    # 2. Run initialization check
    initialize_vector_store_if_empty(my_vector_store, my_embedding_manager, pdf_directory="data/pdf_documents")
    
    # 3. Setup retriever
    my_rag_retriever = RAGRetriever(my_vector_store, my_embedding_manager)
    
    # 4. Setup Groq LLM client
    groq_secret_api_key = os.getenv("GROQ_API_KEY")
    if not groq_secret_api_key:
        print("ERROR: GROQ_API_KEY is missing from environment variables!")
    
    my_chat_groq_llm = ChatGroq(
        groq_api_key=groq_secret_api_key,
        model_name="openai/gpt-oss-20b",
        temperature=0.1
    )
    
    # 5. Interactive terminal question test
    user_test_question = input("Ask any question related to your cooking PDFs: ")
    
    final_answer_output = rag_simple(user_test_question, my_rag_retriever, my_chat_groq_llm)
    
    print("FINAL ANSWER:")
    print(final_answer_output)