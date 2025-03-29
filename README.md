# AR-EN-Sales-Agent

A bilingual (Arabic/English) AI-powered sales assistant for electronics products using Multi Vector Retriever architecture and retrieval-augmented generation (RAG).

## Overview

AR-EN-Sales-Agent is an intelligent virtual sales assistant that can answer customer queries about electronics products in both Arabic and English. The system leverages the Multi Vector Retriever pattern to efficiently retrieve relevant product information and provide accurate, contextually relevant responses.


![Screenshot 2025-03-29 025819](https://github.com/user-attachments/assets/4b2d2c69-a7a7-4f41-903f-33e0b6917d78)

## How It Works

As illustrated in the architecture diagram above:

1. **Document Processing**: Full product documents (JSON data) are loaded into the system
2. **Summarization**: gemma2-9b-it generates concise summaries of each document
3. **Multi-representation Indexing**:
   - Summaries are embedded and stored in a vector database for efficient semantic search
   - Original documents are kept in a document store
4. **Retrieval Process**:
   - When a question is received, it's embedded and used to search the vector store
   - The most relevant summaries are identified
   - The system retrieves the corresponding full documents from the document store
   - These documents provide the context for generating accurate responses

## Features

- **Bilingual Support**: Automatically detects and responds in Arabic or English
- **Multi Vector Retrieval**: Implements the efficient two-stage retrieval system shown in the diagram
- **Conversation Memory**: Maintains chat history for contextual responses
- **API Interface**: Exposes functionality through a FastAPI endpoint
- **Efficient Processing**: Uses quantized models for improved performance


## Models Used

This project leverages several powerful AI models:

1. **Gemma2-9b-it** (via Groq): Used for generating concise document summaries
2. **BAAI/bge-m3**: Embedding model for vector representation of documents
3. **CohereForAI/aya-expanse-8b**: Main language model for generating responses in both Arabic and English (4-bit quantized)

## Installation

```bash
# Clone the repository
git clone https://github.com/Shrouk-Adel/AR-EN-Sales-Agent.git
cd AR-EN-Sales-Agent

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file with your API keys:
```
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

2. Place your product data JSON file in the appropriate directory

## Usage

### Starting the API server

```bash
uvicorn main:app --reload
```

### Making API requests

```python
import requests
import json

url = "http://localhost:8000/answer"
payload = {"question": "What are the specifications of the latest smartphone?"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json())
```

### Arabic Example

```python
payload = {"question": "ما هي مواصفات أحدث هاتف ذكي؟"}
response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json())
```

## Future Improvements
- Implement user feedback mechanism
- Expand language support
- Add authentication to the API
- Improve response time through further optimization

 
