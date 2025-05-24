# agneyastra

## Setup & Installation

This project provides a multi-agent AI-powered legal research assistant, primarily implemented as a Streamlit app.

### Requirements

•⁠  ⁠Python 3.8+
•⁠  ⁠[Streamlit](https://streamlit.io/)
•⁠  ⁠Other dependencies: ⁠ PyPDF2 ⁠, ⁠ requests ⁠, ⁠ google-generativeai ⁠, ⁠ groq ⁠, ⁠ beautifulsoup4 ⁠, etc.

You can install all dependencies using:

⁠ bash
pip install -r requirements.txt
 ⁠
(If ⁠ requirements.txt ⁠ is missing, install the above libraries manually.)

### Running the App

1.⁠ ⁠*Clone the repository:*

    ⁠ bash
    git clone https://github.com/sameersheikh22/agneyastra.git
    cd agneyastra
     ⁠

2.⁠ ⁠*Install dependencies:*

    ⁠ bash
    pip install streamlit PyPDF2 requests google-generativeai groq beautifulsoup4
     ⁠

3.⁠ ⁠*Run the Streamlit app:*

    ⁠ bash
    streamlit run simple_streamlit_legal_research_assistant.py
     ⁠

4.⁠ ⁠*Configuration:*
   - Enter your Gemini API keys (comma-separated) and Groq API key in the sidebar when prompted.
   - Adjust other agent and research settings as needed.

---

## Multi-Agent Architecture

The core logic is implemented in ⁠ simple_streamlit_legal_research_assistant.py ⁠. The architecture consists of the following agents, each responsible for a specific step in the research workflow:

1.⁠ ⁠*Argument Extraction Agent*
   - Extracts the core thesis, key legal concepts, and new research angle from the user's base paper and critique.
   - Supports Gemini and Ollama as backend models.

2.⁠ ⁠*Keyword Generator Agent*
   - Generates intelligent keyword permutations for efficient legal research, based on extracted arguments and user-provided seed keywords.

3.⁠ ⁠*Source Crawler Agent*
   - Searches multiple sources (e.g., SSRN, JSTOR for scholarly articles; court databases for case law; news sites) using the generated keywords.
   - Aggregates and deduplicates results.

4.⁠ ⁠*Citation Chainer Agent*
   - Analyzes top papers to suggest related works, landmark cases, and seminal articles for citation chaining.

5.⁠ ⁠*Summary Extraction & Case Analysis Agents*
   - Extracts concise summaries and key findings from papers.
   - For case law, analyzes facts, legal issues, arguments, judgments, and court findings.

6.⁠ ⁠*Relevance Scorer Agent*
   - Scores and ranks all results according to their relevance to the new research angle and key concepts.

### Main Workflow

•⁠  ⁠Users input a base paper (as text or URL) and specify a research angle.
•⁠  ⁠The agents process the input stepwise: extracting arguments → generating keywords → crawling sources → chaining citations → extracting summaries/analyses → scoring relevance.
•⁠  ⁠Results are displayed interactively, with options to download research briefs or export data.

---

## Notes

•⁠  ⁠All logic is contained in the root Streamlit app file: [⁠ simple_streamlit_legal_research_assistant.py ⁠](simple_streamlit_legal_research_assistant.py)
•⁠