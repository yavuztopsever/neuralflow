# NeuralFlow

NeuralFlow is a playground project where I experiment with the extent of finetuning and training models like LLMs and embedding models with personal chat and documentation data to assess their potential value in personal assistance tasks and chatbots. I aim to leverage the latest approaches, such as LangGraph and graph-based context retrieval, to explore the potential of more complex agentic or non-agentic logic flows.

The project is constantly evolving and experimental, so its primary focus is on education and discovery rather than user-friendliness. Despite this, I strive to maintain its functionality as much as possible. Feel free to experiment with the project or use it as a foundation for your own applications.


## Author

**Yavuz Topsever** - Data Consultant  
Munich, Germany

### Contact & Links
- Email: yavuz.topsever@windowslive.com
- Phone: +49 162 7621469
- [GitHub](https://github.com/yavuztopsever)
- [LinkedIn](https://www.linkedin.com/in/yavuztopsever)

### Technical Expertise
- **Languages**: Python, Java, JavaScript
- **ML/AI**: PyTorch, Scikit-learn, CNNs, RNNs, Transformers
- **NLP/Graphs**: NLTK, LangChain, LangGraph, Graph Neural Networks (GNNs)
- **Backend**: FastAPI, RESTful API Design, SQLite, Redis
- **DevOps**: Docker, CI/CD, Pytest
- **Languages**: English (Fluent), German (Beginner), French (Proficient), Turkish (Native)

## Current Features

- **LangChain Integration**: Built on LangChain 0.1.0+ with support for OpenAI and community models
- **Vector Store Support**: Integration with ChromaDB, FAISS, Pinecone, and Weaviate
- **Advanced LLM Support**: OpenAI API integration with tiktoken for token management
- **Data Processing**: Support for PDF processing, web crawling, and Wikipedia data extraction
- **Visualization Tools**: Built-in support for matplotlib, seaborn, and plotly
- **Database Management**: SQLAlchemy with PostgreSQL support and Alembic migrations
- **Monitoring**: Prometheus metrics and structured logging
- **Development Tools**: Comprehensive testing setup with pytest
- **Model Support**: PyTorch 2.1.0+ integration with Hugging Face Transformers

## Project Structure

```
neuralflow/
├── src/
│   └── neuralflow/
│       ├── frontend/     # Frontend components
│       ├── infrastructure/ # Core infrastructure
│       ├── tools/        # Utility tools
│       ├── config.py     # Configuration management
│       ├── main.py      # Application entry point
│       └── __init__.py  # Package initialization
├── models/          # Model files and data
├── tests/           # Test suite
├── docs/           # Documentation
├── scripts/        # Utility scripts
├── .env            # Environment configuration
├── requirements.txt # Production dependencies
├── requirements-dev.txt # Development dependencies
├── pyproject.toml  # Project configuration
├── setup.py        # Package setup
└── pytest.ini      # Test configuration
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yavuztopsever/neuralflow.git
cd neuralflow
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For development, install additional dependencies:
```bash
pip install -r requirements-dev.txt
```

5. Set up configuration:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Quick Start

1. Start the application:
```bash
python -m neuralflow.main
```

2. Access the application interface (URL will be provided in the console output)

## Development

### Running Tests
```bash
pytest
```

### Code Coverage
```bash
pytest --cov=neuralflow tests/
```

## Documentation

Documentation is available in the `docs` directory:

- API Documentation
- Architecture Overview
- Development Guidelines
- Deployment Instructions
- User Guides

## Contributing

Contributions are welcome! Please read our Contributing Guide in the documentation for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support:
1. Check the documentation in the `docs` directory
2. Open an issue on GitHub
3. Contact the author directly through the provided contact information
