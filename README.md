# Healthcare MCP Server

A sophisticated Model Context Protocol (MCP) server that integrates Google's Gemini LLM with advanced medical AI capabilities for healthcare applications. This system provides intelligent medical analysis, diagnosis assistance, and health insights using state-of-the-art machine learning models.

## Features

- 🏥 **Advanced Medical Analysis**
  - Symptom analysis with multi-model approach
  - Medical image processing and classification
  - Evidence-based treatment recommendations
  - Health metrics monitoring and analysis

- 🤖 **AI Integration**
  - Google Gemini LLM integration
  - Medical-specific NLP models
  - Computer vision for medical imaging
  - Specialized medical embeddings

- 👨‍⚕️ **Healthcare Management**
  - Patient and doctor portals
  - Appointment scheduling
  - Medical history tracking
  - Secure authentication system

- 📊 **Health Insights**
  - Personalized health recommendations
  - Risk factor identification
  - Vital signs monitoring
  - Trend analysis and reporting

## Prerequisites

- Python 3.8+
- pip package manager
- Google Gemini API key
- Virtual environment (recommended)
- Storage Requirements:
  - At least 20GB free disk space for AI models
  - Models are cached locally at `C:\Users\<username>\.cache\huggingface\hub`
  - First run will download required models (15-30 minutes)

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows:
   .\venv\Scripts\activate
   
   # On Unix/MacOS:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   - Create a `.env` file in the project root
   - Add required environment variables:
     ```
     GOOGLE_API_KEY=your-gemini-api-key
     SECRET_KEY=your-flask-secret-key
     ```

## Running the Server

1. Initialize the database:
   ```bash
   python server.py --init-db
   ```

2. Start the server:
   ```bash
   python launcher.py
   ```

The server will be available at `http://localhost:5000`

## Command-Line Usage

The server provides several command-line options for managing AI models and running the system:

### Basic Usage

1. Start the server (with automatic model download):
   ```bash
   python launcher.py
   ```

2. Start without downloading missing models:
   ```bash
   python launcher.py --download-models=false
   ```

### Model Management

1. View model information and storage usage:
   ```bash
   python launcher.py --model-info
   ```
   Shows:
   - Total storage used
   - Downloaded models
   - Missing required models

2. Clear model cache:
   ```bash
   # Clear specific model cache:
   python launcher.py --manage-models --clear-cache medical_nlp
   python launcher.py --manage-models --clear-cache vision
   python launcher.py --manage-models --clear-cache medical_bert

   # Clear all model caches:
   python launcher.py --manage-models --clear-cache
   ```

3. Force download missing models:
   ```bash
   python launcher.py --download-models
   ```

### Advanced Options

- `--manage-models`: Enter model management mode
- `--clear-cache MODEL_ID`: Clear cache for specific model
- `--model-info`: Show model storage information
- `--download-models`: Force download of missing models

### Examples

1. Check system status without starting:
   ```bash
   python launcher.py --model-info
   ```

2. Clear all caches and start fresh:
   ```bash
   python launcher.py --manage-models --clear-cache
   python launcher.py --download-models
   ```

3. Minimal startup (cloud-only features):
   ```bash
   python launcher.py --download-models=false
   ```

## Documentation

Comprehensive documentation is available in the `/docs` directory:

### 1. Technical Guide
[Technical Guide](docs/technical_guide.md)
- System architecture and components
- Development setup and guidelines
- Deployment instructions
- Security considerations
- Troubleshooting guide
- Performance monitoring

### 2. API Reference
[API Reference](docs/api_reference.md)
- Complete API endpoints documentation
- Authentication methods
- Request/response formats
- Error handling
- Rate limiting
- API versioning

### 3. User Guide
[User Guide](docs/user_guide.md)
- Getting started guide
- Patient portal instructions
- Doctor portal guide
- Common features walkthrough
- Security best practices
- Emergency procedures

For inline documentation of the codebase:
- Check docstrings in `server.py` for detailed function and class documentation
- Review configuration options in `config/mcp_config.py`
- Examine medical resources in `resources/` directory

## Docker Deployment

### Prerequisites
- Docker installed on your system
- Google Gemini API key

### Building the Docker Image
```bash
docker build -t healthcare-mcp .
```

### Running the Container
1. Create a `.env` file with your credentials:
   ```
   GOOGLE_API_KEY=your-gemini-api-key
   SECRET_KEY=your-flask-secret-key
   ```

2. Run the container:
   ```bash
   docker run -d \
     --name healthcare-mcp \
     -p 5000:5000 \
     --env-file .env \
     -v healthcare_models:/root/.cache/huggingface/hub \
     -v healthcare_data:/app/instance \
     healthcare-mcp
   ```

   This command:
   - Maps port 5000 to your host
   - Loads environment variables from .env
   - Creates persistent volumes for AI models and database
   - Runs the container in detached mode

3. View logs:
   ```bash
   docker logs -f healthcare-mcp
   ```

### Storage Volumes
- `healthcare_models`: Stores downloaded AI models (~20GB)
- `healthcare_data`: Stores SQLite database and application data

### Maintenance
- Restart container: `docker restart healthcare-mcp`
- Stop container: `docker stop healthcare-mcp`
- Remove container: `docker rm healthcare-mcp`
- Remove volumes: `docker volume rm healthcare_models healthcare_data`

## Project Structure

```
.
├── launcher.py              # Application entry point
├── server.py               # Main MCP server implementation
├── requirements.txt        # Project dependencies
│
├── config/
│   └── mcp_config.py      # Configuration settings
│
├── resources/
│   ├── medical_ontology.json    # Medical knowledge base
│   └── prompt_templates.json    # LLM prompt templates
│
├── templates/              # Web interface templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── doctor_dashboard.html
│   └── patient_dashboard.html
│
└── tools/                 # Specialized medical AI tools
    ├── medical_nlp.py     # Natural language processing
    ├── medical_imaging.py # Image analysis
    └── health_insights.py # Health metrics analysis
```

## Available Tools

### 1. Medical Analysis Tools

#### Symptom Analysis
```python
analyze_symptoms(symptoms: List[str], ctx: Context, patient_data: Optional[dict] = None) -> Dict
```
- Analyzes symptoms using medical NLP models
- Provides potential diagnoses with confidence levels
- Assesses urgency and recommends immediate actions

#### Medical Image Analysis
```python
medical_image_analysis(image_path: str, ctx: Context) -> Dict
```
- Processes medical images using computer vision
- Supports various medical imaging formats
- Provides detailed analysis and classifications

#### Treatment Suggestions
```python
get_treatment_suggestions(condition: str, patient_history: str, ctx: Context) -> Dict
```
- Generates evidence-based treatment plans
- Considers patient history and context
- Provides alternative treatment options

### 2. Health Insight Tools

#### Health Metrics Analysis
```python
generate_health_insights(patient_data: Dict, ctx: Context) -> Dict
```
- Analyzes patient health metrics
- Identifies risk factors
- Provides personalized recommendations

### 3. LLM Integration

#### Text Generation
```python
generate_text(prompt: str, ctx: Context) -> str
```
- Integrates with Gemini LLM
- Handles medical context appropriately
- Implements safety filters

## Security Features

- Secure authentication system
- Role-based access control
- Password hashing
- Session management
- Environment variable protection

## Medical Data Resources

The system uses comprehensive medical resources:

1. **Medical Ontology**
   - Symptom categories
   - Disease classifications
   - Treatment protocols
   - Risk assessment guidelines

2. **Prompt Templates**
   - Diagnosis templates
   - Treatment planning
   - Patient education
   - Health monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For support and questions, please open an issue in the repository.

## Acknowledgments

- Google Gemini AI
- Medical NLP community
- Healthcare AI researchers
- Open-source medical datasets

## Version History

- v1.0.0 - Initial release
- v1.1.0 - Added medical imaging support
- v1.2.0 - Enhanced health insights
- v1.3.0 - Improved NLP capabilities