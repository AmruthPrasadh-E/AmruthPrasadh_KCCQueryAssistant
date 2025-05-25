# KCC Chatbot

![KCC Chatbot Logo](logo.jpg)

## Overview

KCC Chatbot is an intelligent agricultural assistant based on the Kisan Call Centre (KCC) knowledge base. The chatbot provides farmers and agricultural enthusiasts with reliable information about farming practices, crop management, pest control, and other agricultural topics.

The system implements a hybrid search approach:
1. First attempts to answer questions using the local KCC knowledge base
2. Falls back to internet search via Tavily API when local context is insufficient

## Features

- **Semantic Search**: Uses FAISS vector store for efficient similarity search to find relevant information
- **Hybrid Knowledge Source**: Combines local KCC database with real-time internet search capabilities
- **Context-Aware Responses**: Generates responses based on retrieved context using LLM
- **User-Friendly Interface**: Simple web interface built with Streamlit

## Project Structure

```
KCC_Chatbot/
├── .env                    # Environment variables configuration
├── app.py                  # Streamlit web application
├── kcc_chat.py             # Core chatbot functionality
├── requirements.txt        # Project dependencies
├── logo.jpg                # Project logo
├── vector_base/            # Vector store creation and management
│   ├── data_cleaning.py    # Data preprocessing utilities
│   └── build_vector_store.py # Vector store building utilities
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/KCC_Chatbot.git
   cd KCC_Chatbot
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables by creating a `.env` file:
   ```
   OLLAMA_LLM_MODEL="mistral"
   OLLAMA_EMBEDDING_MODEL="mxbai-embed-large"
   TAVILY_API_KEY="your_tavily_api_key"
   ```

## Data Preparation

1. Data cleaning:
   ```
   python vector_base/data_cleaning.py
   ```
   This script processes the KCC dataset and creates a cleaned version with only the relevant columns.

2. Building the vector store:
   ```
   python vector_base/build_vector_store.py
   ```
   This creates a FAISS vector store from the cleaned KCC dataset for efficient semantic search.

## Usage

1. Start the Streamlit web application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Enter your agricultural query in the text input field and submit

4. The chatbot will respond with relevant information from the KCC knowledge base or from the internet if no local information is available

## Core Components

### Query Handling

The query handling process follows these steps:
1. User submits a query through the UI
2. System retrieves top-k context via semantic search from the FAISS vector store
3. If relevant context is found:
   - The context and query are passed to the LLM to generate a response
4. If no relevant context is found:
   - The system notifies that no local context was found
   - It performs a live internet search via Tavily API
   - Displays fallback results from the internet

### Vector Store

The vector store is built using:
- FAISS for efficient similarity search
- Ollama embeddings for converting text to vector representations
- Processed KCC dataset containing agricultural questions and answers

## Dependencies

- langchain: Framework for building LLM applications
- streamlit: Web application framework
- faiss-cpu: Vector similarity search library
- langchain-ollama: Integration with Ollama models
- langchain-tavily: Integration with Tavily search API
- python-dotenv: Environment variable management

## Requirements

- Python 3.8+
- Ollama running locally with the specified models
- Tavily API key for internet search capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kisan Call Centre (KCC) for the agricultural knowledge base
- Ollama for providing local LLM capabilities
- Tavily for the search API
