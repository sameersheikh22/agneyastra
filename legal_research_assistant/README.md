# Multi-Agent Legal Research Companion

This project aims to build an AI-powered multi-agent research system to help law students find relevant supporting material for their research.

## Project Status

This project is currently in the initial setup phase. The core agent functionalities are placeholders.

## Architecture Overview

The system uses a multi-agent approach:
- **ArgumentExtractionAgent**: Understands the base paper and research angle.
- **KeywordGeneratorAgent**: Creates search terms.
- **SourceCrawlerAgent**: Searches scholarly sources (SSRN, JSTOR).
- **CitationChainerAgent**: Follows citation trails.
- **RelevanceScorerAgent**: Scores sources for relevance.
The orchestration of these agents is handled in `main.py`.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd legal-research-assistant
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **API Key Configuration:**
    This project requires API keys for various services (Google AI, Groq, Tavily, Serper).
    - Create a file named `config.py` in the `legal_research_assistant` directory.
    - Copy the contents of `config.example.py` (you will create this in a later step, for now, just note this) into `config.py`.
    - Add your API keys to `config.py`:
      ```python
      API_KEYS = {
          "GOOGLE_AI_API_KEY": "YOUR_GOOGLE_AI_KEY_HERE",
          "GROQ_API_KEY": "YOUR_GROQ_KEY_HERE",
          "TAVILY_API_KEY": "YOUR_TAVILY_KEY_HERE",
          "SERPER_DEV_API_KEY": "YOUR_SERPER_DEV_KEY_HERE"
      }
      ```
    **Important**: `config.py` should not be committed to version control if it contains real keys. Ensure it's listed in `.gitignore`.

## Running the Application (Current State)

To run the application (with placeholder functionality):
```bash
python legal_research_assistant/main.py
```

This will execute the basic orchestration logic, but as the agents are placeholders, it will not perform real research yet.
