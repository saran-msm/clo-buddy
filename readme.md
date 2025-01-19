# CLO Buddy - Legal Document Analysis System

## Overview
CLO Buddy is an advanced legal document analysis system designed to help legal professionals efficiently process, analyze, and extract insights from legal documents. The system leverages state-of-the-art language models and vector database technology to provide accurate summaries, key highlights, and actionable insights.

## Features
- **Document Processing**
  - PDF and text file support
  - Automatic text extraction
  - Intelligent text chunking
  - Vector-based document storage

- **Analysis Capabilities**
  - Document summarization
  - Key highlight extraction
  - Legal reference identification
  - Actionable insights generation

- **Security & Compliance**
  - PII detection and anonymization
  - Content filtering
  - Input validation
  - Response sanitization

- **Advanced Search**
  - Semantic search functionality
  - Vector-based similarity search
  - Metadata filtering
  - Contextual document retrieval

- **Quality Assurance**
  - ROUGE score evaluation
  - Perplexity measurement
  - Response validation
  - Performance metrics

## Technology Stack
- **Backend Framework**
  - Flask (Python)
  - ThreadPoolExecutor for concurrent processing

- **Language Models**
  - FLAN-T5-Base for text generation
  - GPT-4 for summarization
  - MiniLM-L6-v2 for embeddings
  - GPT-2 for perplexity calculation

- **Vector Database**
  - ChromaDB for persistent storage
  - Sentence Transformers for embeddings
  - Vector similarity search

- **Document Processing**
  - PyPDF2 for PDF extraction
  - Custom text chunking
  - UTF-8 text processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/clo-buddy.git
cd clo-buddy
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p data/vectordb
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Start the server:
```bash
python app.py
```

2. Access the web interface:
```
http://localhost:5001
```

3. Upload a legal document and wait for analysis results.

## API Endpoints

### Document Processing
```http
POST /process
Content-Type: multipart/form-data
Body: file=@document.pdf
```

### Semantic Search
```http
POST /search
Content-Type: application/json
Body: {
    "query": "your search query"
}
```

## Project Structure
```
clo-buddy/
├── app.py                 # Main Flask application
├── requirements.txt       # Project dependencies
├── data/                 # Data storage
│   └── vectordb/        # Vector database storage
├── static/              # Static files
│   ├── css/            # Stylesheets
│   └── js/             # JavaScript files
├── templates/           # HTML templates
└── utils/              # Utility modules
    ├── document_processor.py  # Document processing logic
    ├── evaluation.py         # Evaluation metrics
    ├── guardrails.py         # Security measures
    └── vector_store.py       # Vector database operations
```

## Configuration

Key configuration options in `.env`:
```env
OPENAI_API_KEY=your_api_key_here
MAX_FILE_SIZE=16777216  # 16MB
ALLOWED_EXTENSIONS=pdf,txt
```

## Evaluation Metrics

The system provides several quality metrics:
- ROUGE scores (Precision, Recall, F1)
- Perplexity scores for fluency
- Processing time metrics
- Accuracy measurements

## Security Measures

- File type validation
- Size restrictions
- PII detection
- Content filtering
- Input sanitization
- Response validation
