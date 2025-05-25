"""Vector Store Builder Module for KCC Chatbot

This module provides functionality to:
1. Build a FAISS vector store from cleaned KCC dataset CSV files
2. Perform test queries on the vector store with semantic search capabilities

The module uses Ollama's embedding model to convert text into vector embeddings,
and FAISS for efficient similarity search. It processes data in chunks to manage
memory usage when dealing with large datasets.

Main components:
- build_vector_store: Creates and saves a FAISS vector store from CSV data
- query_vector_store: Retrieves semantically similar documents for a query

Note: For development purposes, only the first chunk of data is processed.
"""

import pandas as pd
from tqdm import tqdm  # For progress bars during processing
from langchain_ollama.embeddings import OllamaEmbeddings  # For text-to-vector embedding
from langchain_community.vectorstores import FAISS  # For vector similarity search
from langchain_core.documents import Document  # For document representation
import logging
import os
from dotenv import load_dotenv

# Configure logging to track vector store operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='vector_store.log', filemode='a')
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize embedding model
embedding_model = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDING_MODEL"),
)

def build_vector_store(csv_path, output_path):
    """
    Create a FAISS vector store from the cleaned KCC dataset CSV file.
    
    This function:
    1. Loads the cleaned CSV data in manageable chunks
    2. Creates document objects with question-answer pairs
    3. Generates vector embeddings for each document
    4. Builds a FAISS index for efficient similarity search
    5. Saves the vector store to disk for later use
    
    For development purposes, only processes the first chunk of data.
    
    Args:
        csv_path (str): Path to the cleaned CSV file containing QueryText and KccAns columns
        output_path (str): Directory path where the FAISS vector store will be saved
        
    Returns:
        bool: True if vector store creation was successful, False otherwise
    """
    logger.info(f"Loading data from {csv_path}")
    
    # Set chunk size for processing large CSV files in memory-efficient manner
    chunk_size = 20000  # Process 20,000 rows at a time to avoid memory issues
    
    # Initialize empty list to store document objects for vector store creation
    all_documents = []
    
    # Process the CSV file in chunks with a progress bar
    for chunk_num, chunk in enumerate(tqdm(pd.read_csv(csv_path, chunksize=chunk_size), desc="Processing CSV chunks")):
        
        logger.info(f"Processing chunk {chunk_num+1}")
        # Extract only the question and answer columns from the dataset
        df = chunk[['QueryText', 'KccAns']]
        
        # Convert each row into a Document object for the vector store
        for i, row in df.iterrows():
            question = row['QueryText']  # User's question
            kcc_ans = row['KccAns']      # KCC's answer to the question
            
            # Format the document content with clear question-answer structure
            # This format helps the retrieval system understand context
            chunk_text = f"Question: {question}\nKCC response: {kcc_ans}"
            
            # Create a Document object with the formatted text and metadata
            # Metadata is preserved for reference and filtering capabilities
            doc = Document(
                page_content=chunk_text,  # The formatted Q&A text
                metadata={
                    "Question": question,     # Original question for reference
                    "KCC Response": kcc_ans  # Original answer for reference
                }
            )
            
            # Add the document to our collection
            all_documents.append(doc)
        
        # Development mode: only process the first chunk to save time
        # Remove this break statement in production to process all data
        if chunk_num == 0:
            break
    
    logger.info(f"Created {len(all_documents)} document objects")
    
    # Create a new FAISS vector store with the prepared documents
    # FAISS provides efficient similarity search in vector spaces
    logger.info("Creating new vector store...")
    new_vectorstore = FAISS.from_documents(
        all_documents,      # List of Document objects to index
        embedding_model     # The embedding model to convert text to vectors
    )
    
    # Save the vector store to disk for later use
    logger.info(f"Saving vector store to {output_path}")
    new_vectorstore.save_local(output_path)  # Persists the FAISS index and metadata
    
    logger.info("Vector store created and saved successfully!")
    return True  # Indicate successful completion

def query_vector_store(vector_store_path, query_text, k=3):
    """
    Query the FAISS vector store to find semantically similar documents.
    
    This function:
    1. Loads the previously created FAISS vector store
    2. Converts the query text to a vector embedding
    3. Performs similarity search to find the most relevant documents
    4. Returns the top-k matching documents with their similarity scores
    
    Args:
        vector_store_path (str): Path to the directory containing the FAISS vector store
        query_text (str): The user's query to search for
        k (int): Number of top results to return (default: 3)
        
    Returns:
        list: List of tuples, each containing (Document, score) pairs where:
            - Document is a langchain Document object with content and metadata
            - score is a float representing distance (lower is more similar)
    """
    logger.info(f"Loading vector store from {vector_store_path}")
    
    # Create embedding model
    embedding_model = OllamaEmbeddings(
        model="mxbai-embed-large",
    )
    
    # Load the previously saved FAISS vector store from disk
    vectorstore = FAISS.load_local(
        vector_store_path,                    # Path to the stored FAISS index
        embeddings=embedding_model,           # Embedding model for query conversion
        allow_dangerous_deserialization=True  # Required for loading custom objects
    )
    
    # Perform semantic similarity search with the query
    # Returns documents sorted by relevance (most similar first) with similarity scores
    results = vectorstore.similarity_search_with_score(query_text, k=k)
    
    logger.info(f"Found {len(results)} results for query: '{query_text}'")
    return results

if __name__ == "__main__":
    # Main execution block for when this script is run directly
    
    # Path to the cleaned KCC dataset
    csv_path = r"C:\Users\Amruth\Documents\KCC_Chatbot\new_cleaned_kcc_dataset.csv"
    
    # Directory where the vector store will be saved
    output_path = "KCC_vector_store"   
    
    logger.info("Starting vector store creation process")
    
    # Build the vector store from the cleaned data
    success = build_vector_store(csv_path, output_path)
    
    if success:
        # If vector store creation was successful, test it with sample queries
        logger.info("\nTesting the rebuilt vector store:")
        try:
            # Sample agricultural queries to test the vector store's retrieval capabilities
            test_queries = [
                "How to control pests in apple trees?",
                "What is the best fertilizer for tomatoes?",
                "How to increase crop yield?"
            ]
            
            # Test each query and log the results
            for query in test_queries:
                logger.info(f"\nQuery: {query}")
                # Retrieve top 3 most relevant documents for each query
                results = query_vector_store(output_path, query, k=3)
                
                if results:
                    # Log the retrieved documents and their similarity scores
                    logger.info(f"Found {len(results)} relevant documents:")
                    for i, (doc, score) in enumerate(results):
                        # Lower score means higher similarity in FAISS (L2 distance)
                        logger.info(f"\nResult {i+1} (Score: {score:.4f}):")
                        logger.info(f"Content: {doc.page_content}")
                        logger.info(f"Metadata: {doc.metadata}")
                else:
                    logger.info("No relevant documents found.")
                    
        except Exception as e:
            # Handle and log any errors that occur during testing
            logger.error(f"Error testing rebuilt vector store: {e}")
            # Print full stack trace for debugging
            import traceback
            traceback.print_exc()
            print(f"Error testing vector store: {str(e)}")
    else:
        # Log failure if vector store creation was unsuccessful
        logger.error("Failed to rebuild vector store.")
        print("Vector store creation failed. Check the logs for details.")
