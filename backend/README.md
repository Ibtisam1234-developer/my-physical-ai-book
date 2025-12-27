# Physical AI & Humanoid Robotics Backend

This is the backend component of the Physical AI & Humanoid Robotics platform, implementing a RAG (Retrieval-Augmented Generation) chatbot system for humanoid robots.

## Architecture Overview

```
backend/
├── app/                    # Main application code
│   ├── api/               # API endpoints (chat, documents, etc.)
│   ├── config/            # Configuration and settings
│   ├── db/                # Database models and connections
│   ├── rag/               # RAG pipeline (retrieval, embeddings, etc.)
│   ├── schemas/           # Pydantic models
│   └── main.py            # FastAPI application entry point
├── tests/                 # Test files
│   ├── conftest.py        # Test configuration
│   └── test_main.py       # Main application tests
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not committed)
└── .env.example           # Example environment variables
```

## Core Components

### 1. RAG Pipeline (`app/rag/`)
- **Embeddings**: Generate vector representations using Gemini API
- **Ingestion**: Process documents and store in Qdrant vector database
- **Retrieval**: Retrieve relevant context for user queries
- **Prompts**: System prompts for AI generation

### 2. API Endpoints (`app/api/`)
- `/chat`: Main chat endpoint with streaming support
- `/chat/stream`: Server-Sent Events for streaming responses
- `/documents`: Document management and upload

### 3. Database Models (`app/db/`)
- SQLAlchemy async models for document management
- User authentication models
- Chat session models

## Setup Instructions

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA support (for Isaac integration)
- Access to Gemini API key

### Installation

1. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your actual values
```

4. **Run the application:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Environment Variables

Create a `.env` file with the following variables:

```env
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
GEMINI_EMBEDDING_MODEL=text-embedding-004
GEMINI_GENERATION_MODEL=gemini-2.0-flash-exp

# Qdrant Vector Database Configuration
QDRANT_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=physical_ai_docs

# JWT Authentication
JWT_SECRET=supersecretkeyfordevelopment
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
DATABASE_URL=sqlite+aiosqlite:///./physical_ai.db
```

## API Endpoints

### Chat Endpoint
```
POST /api/chat
Content-Type: application/json

{
  "query": "What is Physical AI?",
  "session_id": "optional_session_id"
}
```

### Streaming Chat Endpoint
```
POST /api/chat/stream
Content-Type: application/json

{
  "query": "Explain humanoid locomotion",
  "session_id": "optional_session_id"
}
```

### Document Upload
```
POST /api/documents/upload
Content-Type: multipart/form-data

File: document.pdf
```

## Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_main.py -v
```

## Key Features

- **GPU-Accelerated Inference**: Leverages NVIDIA GPUs for fast processing
- **Real-time Streaming**: Server-Sent Events for live response streaming
- **Document Ingestion**: Support for PDF, MDX, DOCX, and other formats
- **Vector Storage**: Qdrant for efficient similarity search
- **Multi-modal AI**: Integration with Gemini for vision-language-action
- **Authentication**: JWT-based user authentication
- **Rate Limiting**: Protection against API abuse

## Integration with Frontend

This backend serves as the API layer for the Docusaurus-based frontend. The frontend communicates with these endpoints to provide:

- Natural language chat interface
- Document upload and management
- Session persistence
- Real-time response streaming

## Performance Optimization

- **Async Processing**: All endpoints use async/await for high concurrency
- **GPU Acceleration**: Embedding generation uses GPU when available
- **Caching**: Frequently accessed data is cached
- **Batch Processing**: Multiple documents processed in batches

## Troubleshooting

### Common Issues

1. **Gemini API Key Issues**
   - Ensure your API key is valid and has proper permissions
   - Check that the billing account is active

2. **Qdrant Connection Issues**
   - Verify URL and API key are correct
   - Check that the collection exists

3. **CUDA/GPU Issues**
   - Ensure CUDA drivers are properly installed
   - Verify GPU is detected by PyTorch

### Performance Tuning

- Adjust batch sizes based on available GPU memory
- Configure connection pooling for database
- Optimize embedding dimensions for your use case

## Security Considerations

- Never commit `.env` files to version control
- Use HTTPS in production deployments
- Implement proper input validation
- Sanitize user inputs to prevent injection attacks
- Regular security updates for dependencies

## Deployment

For production deployment:

1. Use a production WSGI server (e.g., Gunicorn with Uvicorn workers)
2. Set `ENVIRONMENT=production` in environment variables
3. Use a proper database (PostgreSQL recommended)
4. Implement proper logging and monitoring
5. Set up SSL certificates for HTTPS

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests to ensure everything passes
6. Submit a pull request

## License

This project is licensed under the Apache 2.0 License.

---

*This backend powers the Physical AI & Humanoid Robotics educational platform, providing RAG capabilities for AI-powered learning assistance.*