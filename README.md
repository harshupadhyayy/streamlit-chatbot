# Streamlit Chatbot Project

This repository contains a chatbot application built using [Streamlit](https://streamlit.io/), [Langchain](https://langchain.com/), and [LlamaIndex](https://llamaindex.ai/) for building interactive conversational agents. The project is set up with [Poetry](https://python-poetry.org/) for dependency management and package isolation.


## Features

- **Single-modal chatbot (SimpleChat.py):** A simple text-based chatbot powered by OpenAI's language model via Langchain.
- **Multi-modal chatbot (MultimodalChat.py):** This bot retrieves text and images from Wikipedia and answers user questions based on them.
- **Document ingestion (IngestDocuments.py):** Allows users to upload PDFs for text extraction and indexing using FAISS for vector-based search.
  
## Getting Started

### Prerequisites

- Python 3.8 or later
- Poetry for package management (`pip install poetry`)
- API keys for OpenAI and Qdrant (store these in Streamlit secrets)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/streamlit-chatbot.git
   cd streamlit-chatbot
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

3. **Activate the Poetry environment:**
   ```bash
   poetry shell
   ```

4. **Install additional dependencies using `requirements.txt` for Streamlit:**
   ```bash
   pip install -r requirements.txt
   ```

### Streamlit Secrets

Create a `.streamlit/secrets.toml` file to store your API keys:
```toml
[secrets]
OPENAI_API_KEY = "your-openai-api-key"
QDRANT_CLUSTER_URL = "your-qdrant-url"
QDRANT_API_KEY = "your-qdrant-api-key"
```

### Running the App

To start the Streamlit application, run:

```bash
streamlit run Home.py
```

### Navigating the App

- **Home page (`Home.py`):** Welcome page of the chatbot.
- **Single-modal Chatbot (`SimpleChat.py`):** A simple chatbot that handles text-based conversations.
- **Multi-modal Chatbot (`MultimodalChat.py`):** Allows users to search Wikipedia for relevant text and images, and retrieve answers based on this context.
- **Document Ingestion (`IngestDocuments.py`):** Upload and process PDF documents for later retrieval in the chatbot.

## Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests.

## License

This project is licensed under the MIT License.