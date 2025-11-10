# 🧠 AURA – AI Unified Response Agent  
### **Intelligent Document Processing & Workflow Automation Platform**

AURA is an **enterprise-grade multimodal AI platform** that automates document processing, data extraction, and business workflows.  
Unlike generic chatbots, AURA combines **multi-agent orchestration**, **persistent vector memory**, and **automation pipelines** to process 1000+ documents, apply business logic, and integrate with existing enterprise systems.

---

## 🎯 Key Differentiators

**Why AURA vs ChatGPT or Generic AI Assistants?**

| Feature                   | Generic AI Chatbots        |      AURA                              |
|---------------------------|----------------------------|----------------------------------------|
| **Document Memory**       | Forgets after conversation | Persistent storage in vector database  |
| **Batch Processing**      | One document at a time     | Process 100+ documents simultaneously  |
| **Workflow Automation**   | Manual operations only     | Automated multi-step workflows         |
| **Business Intelligence** | No historical context      | Learns from past documents and patterns|
| **Integration**           | Limited APIs               | Full FastAPI + Supabase + Airflow      |
| **Custom Logic**          | Generic responses          | Company-specific rules and policies    |
| **Scale**                 | Chat-based limits          | Production-ready batch processing      |

AURA isn’t just an assistant — it’s a **backend automation system** that learns, acts, and scales.

---

## ⚙️ Core Features
### Multi-Modal Document Processing
- **Text Extraction**: PDF, DOCX, TXT files with intelligent parsing
- **OCR**: Extract text from scanned documents and images
- **Audio Transcription**: Convert speech to text with OpenAI Whisper
- **Image Analysis**: Visual understanding with GPT-4 Vision

### Intelligent Automation
- **Smart Routing**: Automatically classify and route documents to specialized agents
- **Data Extraction**: Pull structured data from unstructured documents
- **Validation**: Apply business rules and flag anomalies
- **Batch Processing**: Handle hundreds of documents in parallel

### Enterprise Features
- **Vector Search**: Semantic search across millions of documents using Pinecone
- **Persistent Memory**: Never lose context - store and query entire document history
- **Audit Trail**: Track all document processing with Supabase storage
- **API-First**: RESTful API for easy integration with existing systems

## 💼 Use Cases
- **Invoice Processing:** Automate data extraction and validation across 100s of invoices.
- **Resume Screening:** Parse and rank resumes using embeddings and custom filters.
- **Contract Analysis:** Compare clauses, flag risks, and ensure policy compliance.

## 🧠 Tech Stack

### AI & Machine Learning
| Component | Technology | Purpose |
|-----------|-------------|----------|
| **LLMs** | OpenAI GPT-4o-mini (text), GPT-4.1-mini (vision) | Natural language understanding and multimodal reasoning |
| **Speech-to-Text** | OpenAI Whisper | Audio transcription and voice query processing |
| **Frameworks** | LangChain + LangGraph | Multi-agent orchestration and RAG pipeline |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Semantic vectorization for document search |
| **RAG** | Custom Retrieval-Augmented Generation | Contextual information retrieval and reasoning |

---

### Infrastructure
| Component | Technology | Purpose |
|-----------|-------------|----------|
| **Vector Database** | Pinecone | Semantic search, long-term vector memory |
| **Storage** | Supabase | File management and metadata storage |
| **Database** | PostgreSQL (via Supabase) | Structured data persistence |
| **Message Queue** | RabbitMQ | Asynchronous inter-agent communication |
| **Workflow Orchestration** | Apache Airflow | Batch and scheduled workflow automation |

---

### Backend & API
| Component | Technology | Purpose |
|-----------|-------------|----------|
| **API Framework** | FastAPI | RESTful API endpoints for integration |
| **UI Framework** | Gradio | Interactive multimodal user interface |
| **Authentication** | OAuth2 + JWT | Secure access control for API and UI |

---

### DevOps & Monitoring
| Component | Technology | Purpose |
|-----------|-------------|----------|
| **Containerization** | Docker + Docker Compose | Environment consistency and easy deployment |
| **CI/CD Pipeline** | GitHub Actions | Automated build, test, and deploy workflow |
| **Monitoring** | Prometheus + Grafana | Real-time system metrics and observability |
| **Logging** | Structured Logging (RotatingFileHandler) | Persistent and searchable application logs |


## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.9+ ([Download](https://www.python.org/downloads/))
- Docker & Docker Compose ([Download](https://www.docker.com/))
- Git ([Download](https://git-scm.com/))

### Required API Keys

You'll need accounts and API keys for:
- [OpenAI](https://platform.openai.com/) - For LLM capabilities
- [Pinecone](https://www.pinecone.io/) - For vector database
- [Supabase](https://supabase.com/) - For storage and database

## Installation

### 1. Clone the Repository

```
git clone https://github.com/deepmehta27/AURA---AI-Unified-Response-Agent.git
cd AURA---AI-Unified-Response-Agent
```

### 2. Create Virtual Environment

```
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```
# Copy example environment file
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or vim .env, or code .env
```

Required environment variables:
```
OPENAI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_ENVIRONMENT=your_env_here
SUPABASE_URL=your_url_here
SUPABASE_KEY=your_key_here
```

### 5. Start Infrastructure Services

```
# Start RabbitMQ, Prometheus, and Grafana
cd docker
docker-compose up -d
cd ..
```

## Usage

### Start the FastAPI Backend

```
cd api
uvicorn main:app --reload --port 8000
```

API will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### Start the Gradio UI

```
cd ui
python app.py
```

UI will be available at: `http://localhost:7860`

### Access Monitoring Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **RabbitMQ Management**: http://localhost:15672 (guest/guest)

## 🧱 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                       │
│                   (Gradio UI / FastAPI)                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 Multi-Agent Orchestrator                    │
│              (Coordinates Agent Activities)                 │
└──────┬──────────────────┬──────────────────┬────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Text Agent  │    │Image Agent  │    │Audio Agent  │
│             │    │             │    │             │
│ -  NLP      │    │ -  OCR      │    │ -  STT      │
│ -  QA       │    │ -  Analysis │    │ -  Analysis │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┴──────────────────┘
                         │
                         ▼
       ┌──────────────────────────────────────┐
       │      RAG Pipeline & Vector Search    │
       │         (Pinecone + LangChain)       │
       └──────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌────────────┐  ┌─────────────┐  ┌──────────────┐
│  Supabase  │  │  RabbitMQ   │  │  PostgreSQL  │
│  Storage   │  │  Messaging  │  │   Database   │
└────────────┘  └─────────────┘  └──────────────┘
```

### Key Components

1. **User Interface Layer**: Gradio UI and FastAPI for user interaction
2. **Agent Orchestration**: Coordinates multiple specialized agents
3. **Processing Agents**: Text, Image, and Audio agents handle specific modalities
4. **RAG Pipeline**: Retrieval-Augmented Generation for accurate responses
5. **Infrastructure**: Pinecone, Supabase, RabbitMQ, PostgreSQL

## Project Structure

```
aura/
├── agents/             # AI agent implementations
├── api/                # FastAPI backend
├── ui/                 # Gradio interface
├── utils/              # Helper utilities
├── models/             # Database models
├── config/             # Configuration files
├── airflow/            # Workflow orchestration
├── monitoring/         # Prometheus metrics
├── docker/             # Docker configurations
├── .github/workflows/  # CI/CD pipelines
└── tests/              # Unit and integration tests
```

## Testing

Run unit tests:

```
pytest tests/ -v
```

## Configuration

Edit `config/config.yaml` to customize:

- LLM model selection
- Agent behavior
- Database connections
- Monitoring settings

## Monitoring

Access monitoring dashboards:

1. **Grafana**: Visualize metrics and create custom dashboards
2. **Prometheus**: Query raw metrics and set up alerts
3. **Application Logs**: Check `logs/` directory for detailed logs

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Deep Mehta**

- GitHub: [@deepmehta27](https://github.com/deepmehta27)
- LinkedIn: [Deep Mehta](https://linkedin.com/in/deepmehta27)
- Email: deepmehta827@gmail.com

## Acknowledgments

- OpenAI for GPT models and Whisper
- Pinecone for vector database
- Supabase for backend infrastructure
- LangChain community for agent frameworks
- FastAPI and Gradio communities for open-source support
- Prometheus for real-time observability tooling

---

Made with ❤️ by Deep Mehta
