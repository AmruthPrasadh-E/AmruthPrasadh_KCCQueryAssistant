"""KCC Chatbot Core Module

This module provides the core functionality for the KCC (Kisan Call Centre) Chatbot,
including query handling, context retrieval, and response generation.

The module implements a hybrid search approach that:
1. First attempts to answer questions using the local KCC knowledge base
2. Falls back to internet search via Tavily API when local context is insufficient

Main components:
- Vector store querying: Retrieves relevant agricultural information from FAISS vector store
- Tavily search: Performs internet search for fallback information
- Response generation: Uses LLM to generate contextual responses based on retrieved information

Environment variables (from .env file):
- OLLAMA_LLM_MODEL: The Ollama model to use for response generation 
- OLLAMA_EMBEDDING_MODEL: The Ollama model to use for text embeddings 
- TAVILY_API_KEY: API key for Tavily search service
"""

# Import necessary libraries
from langchain_ollama.embeddings import OllamaEmbeddings  # For text-to-vector embedding
from langchain_community.vectorstores import FAISS  # For vector similarity search
from langchain_core.prompts import ChatPromptTemplate  # For creating LLM prompts
from langchain_core.output_parsers import StrOutputParser  # For parsing LLM outputs
from langchain_ollama.llms import OllamaLLM  # For local LLM inference
from langchain_tavily import TavilySearch  # For internet search capabilities
import os
from dotenv import load_dotenv  # For loading environment variables

# Load environment variables from .env file
load_dotenv()

# Get API key for Tavily search from environment variables
tavily_api = os.getenv("TAVILY_API_KEY")

# Initialize Ollama LLM model for response generation
# Uses the model specified in the OLLAMA_LLM_MODEL environment variable
model = OllamaLLM(model=os.getenv("OLLAMA_LLM_MODEL"))

# Initialize embedding model for vector search
# Uses the model specified in the OLLAMA_EMBEDDING_MODEL environment variable
embedding_model = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDING_MODEL"),
)

def tavily_retrieve_context(query: str) -> str:
    """Retrieve context from the internet using Tavily Search API.
    
    This function performs an internet search using the Tavily API when the local
    knowledge base doesn't have sufficient information to answer a query.
    
    Args:
        query (str): The user's question or search query
        
    Returns:
        str: Concatenated search results from Tavily, formatted as context
             for the LLM prompt
    """
    # Initialize Tavily search with configuration parameters
    tavily_search = TavilySearch(
        max_results=2,        # Limit to 2 results for conciseness
        topic='general',      # Use general topic for broad coverage
        tavily_api_key=tavily_api  # API key from environment variables
    )

    # Execute the search query
    retrieved_docs = tavily_search.invoke({"query": query})
    
    # Format and concatenate the search results
    # Each result is prefixed with "Internet Content:" for clarity in the prompt
    tavily_context = "\n".join(f"Internet Content: {result['content']}" for result in retrieved_docs['results'])
    
    return tavily_context

def query_vector_store(query_text: str, vector_store_path: str = "KCC_vector_store", k: int = 3) -> str:
    """
    Query the FAISS vector store to find semantically similar documents.
    
    This function:
    1. Loads the previously created FAISS vector store
    2. Converts the query text to a vector embedding
    3. Performs similarity search to find the most relevant documents
    4. Extracts and concatenates the content from the top-k matching documents
    
    Args:
        query_text (str): The user's query to search for
        vector_store_path (str): Path to the directory containing the FAISS vector store
                                 Defaults to "KCC_vector_store"
        k (int): Number of top results to return (default: 3)
        
    Returns:
        str: Concatenated content from the top-k most relevant documents
    """
    try:
        # Load the previously saved FAISS vector store from disk
        vectorstore = FAISS.load_local(
            vector_store_path,                    # Path to the stored FAISS index
            embeddings=embedding_model,           # Embedding model for query conversion
            allow_dangerous_deserialization=True  # Required for loading custom objects
        )
        
        # Perform semantic similarity search with the query
        # Returns documents sorted by relevance (most similar first) with similarity scores
        results = vectorstore.similarity_search_with_score(query_text, k=k)
        
        # Extract and concatenate the content from the retrieved documents
        # Each result is a tuple of (Document, score), we only need the Document's page_content
        context = "\n".join([result[0].page_content for result in results])

        return context
    except Exception as e:
        # Handle potential errors (e.g., vector store not found)
        print(f"Error querying vector store: {str(e)}")
        return "No relevant information found in the KCC knowledge base."

def response_to_query_with_KCCbase(query: str) -> str:
    """Generate a response to the user's query using the KCC knowledge base.
    
    This function:
    1. Retrieves relevant context from the vector store
    2. Creates a prompt with the context and query
    3. Sends the prompt to the LLM to generate a response
    
    Args:
        query (str): The user's question about agriculture or farming
        
    Returns:
        str: The generated response based on the KCC knowledge base
    """
    # Create a prompt template for the LLM
    # The prompt instructs the LLM to act as a KCC assistant and answer based on context
    prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful KCC assistant that can answer questions about agriculture and farming. 
    Read the following context: {context}.
    Using the given context, answer the following question in about 60 words. If you donot find relevant information from the context, then simply answer you don't know the answer. 
    Question: {input}
    """
    )

    # Retrieve relevant context from the vector store
    context = query_vector_store(query)
    
    # Create a processing chain: prompt -> LLM -> string parser
    chain = prompt | model | StrOutputParser()
    
    # Generate the response by invoking the chain with the context and query
    kcc_response = chain.invoke({
        "context": context,  # The retrieved KCC knowledge base context
        "input": query      # The user's original query
    })
    
    return kcc_response

def response_from_Tavily(query: str, kcc_response: str) -> str:
    """Generate a final response using KCC knowledge and internet search as fallback.
    
    This function:
    1. Evaluates if the KCC response adequately answers the query
    2. If KCC response is insufficient, retrieves additional context from the internet
    3. Generates a comprehensive response using both sources of information
    
    The function implements the fallback mechanism of the chatbot, ensuring users
    get helpful information even when the local knowledge base is insufficient.
    
    Args:
        query (str): The user's original question
        kcc_response (str): The response generated from the KCC knowledge base
        
    Returns:
        str: The final response, either from KCC or enhanced with internet information
    """
    # Create a prompt template for the LLM to evaluate and enhance the response
    prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that can answer questions about agriculture and farming.
    The user asked the question: {input}
    Read the response given by KCC: {response}.
    If the response given by KCC answers the question, then say that "KCC advises:" and give the response as it is.
    If the response does not answer the question, only then say the database do not have relevant information then use the internet content provided below to answer the question.
    Internet content: {context}.
    Using the given internet content, answer the following question in about 60 words. If you do not find relevant information from the context, then simply answer you don't know the answer. 
    """
    )

    # Retrieve additional context from the internet using Tavily
    tavily_context = tavily_retrieve_context(query)
    
    # Create a processing chain: prompt -> LLM -> string parser
    chain = prompt | model | StrOutputParser()
    
    # Generate the final response by invoking the chain with all available information
    response = chain.invoke({
        "context": tavily_context,  # Internet search results
        "input": query,            # Original user query
        "response": kcc_response   # Response from KCC knowledge base
    })
    return response

def run_response_app(query: str) -> str:
    """Main function to process a user query and generate a comprehensive response.
    
    This function orchestrates the complete query handling process:
    1. First attempts to answer using the local KCC knowledge base
    2. Then evaluates the KCC response and enhances it with internet information if needed
    
    This is the primary entry point for the chatbot's query handling system.
    
    Args:
        query (str): The user's question about agriculture or farming
        
    Returns:
        str: The final comprehensive response to the user's query
    """
    # Step 1: Generate a response using the KCC knowledge base
    kcc_response = response_to_query_with_KCCbase(query)
    
    # Step 2: Evaluate the KCC response and enhance with internet information if needed
    final_response = response_from_Tavily(query, kcc_response)
    
    return final_response