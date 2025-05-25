import io
import os
from dotenv import load_dotenv
import PyPDF2
import streamlit as st
import requests
from typing import List, Dict, Any
import json
import re
from urllib.parse import urlparse, quote
import time
from datetime import datetime
import google.generativeai as genai
from bs4 import BeautifulSoup
from groq import Groq
from google.api_core import exceptions as google_exceptions
import chromadb
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

# Constants
OLLAMA_EMBEDDING_BASE_URL = "https://apaims2.0.vassarlabs.com/ollama2"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Page config
st.set_page_config(
    page_title="Legal Research Companion",
    page_icon="⚖️",
    layout="wide"
)


# Initialize Ollama Embeddings
def init_ollama_embeddings(base_url: str, model_name: str):
    """Initialize OllamaEmbeddings client."""
    try:
        embeddings = OllamaEmbeddings(base_url=base_url, model=model_name)
        st.sidebar.info("Ollama Embeddings client initialized successfully.")
        return embeddings
    except Exception as e:
        st.sidebar.error(f"Ollama Embeddings Init Error: {e}")
        return None


# Initialize API clients
def init_clients(api_key: str):
    """Initialize API clients with provided keys. Gemini API key is mandatory."""
    gemini_model = None
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.sidebar.error(f"Gemini Init Error: {e}")  # Display error in sidebar for visibility
        gemini_model = None

    # Groq
    groq_client = None
    groq_api_key_env = os.getenv("GROQ_API_KEY")
    if groq_api_key_env:
        try:
            groq_client = Groq(api_key=groq_api_key_env)
            # st.sidebar.info("Groq client initialized with key from .env.") # Optional: for debugging
        except Exception as e:
            st.sidebar.warning(f"Groq Init Error (from .env): {e}")
    else:
        st.sidebar.warning("GROQ_API_KEY not found in .env. Groq features may be unavailable.")

    return gemini_model, groq_client


# Agent 1: Argument Extraction Agent (Ollama)
def argument_extraction_agent_ollama(base_paper_content: str, research_angle: str) -> Dict[str, str]:
    """Extract core thesis and identify new research direction using Ollama."""
    ollama_api_url = "https://apaims2.0.vassarlabs.com/ollama1/api/generate"
    prompt = f"""
    Analyze the following legal research paper and the student's new research angle.

    Base Paper Content:
    {base_paper_content[:st.session_state.get('ollama_max_prompt_chars', 3000)]}  # Limit for API

    Student's Research Angle:
    {research_angle}

    Extract and provide a JSON response with these exact keys:
    - "core_thesis": The main argument of the base paper in 2-3 sentences
    - "key_concepts": A list of 3-5 key legal concepts discussed
    - "new_angle": The student's distinguishing research angle
    - "research_directions": A list of 2-3 potential research directions

    Example format:
    {{
        "core_thesis": "The paper argues that...",
        "key_concepts": ["concept1", "concept2", "concept3"],
        "new_angle": "Focus on...",
        "research_directions": ["Direction 1", "Direction 2"]
    }}
    """

    payload = {
        "model": "gemma3:12b",
        "prompt": prompt,
        "stream": False
    }

    advanced_options_str = st.session_state.get('ollama_advanced_options_str', '{}')
    if advanced_options_str and advanced_options_str.strip() != '{}':
        try:
            parsed_options = json.loads(advanced_options_str)
            if isinstance(parsed_options, dict) and parsed_options:
                payload['options'] = parsed_options
            elif parsed_options:
                st.warning(
                    f"Ollama Argument Extraction: Invalid format for advanced options: Expected a non-empty JSON object. Options ignored. Input: {advanced_options_str}")
        except json.JSONDecodeError:
            st.warning(
                f"Ollama Argument Extraction: Could not parse advanced options (invalid JSON): {advanced_options_str}. Proceeding without them.")

    error_response_template = {
        "core_thesis": "Error: Could not extract thesis from Ollama.",
        "key_concepts": [],
        "new_angle": research_angle,
        "research_directions": ["Error: Could not get research directions from Ollama."]
    }

    try:
        response = requests.post(ollama_api_url, json=payload, timeout=60)  # Added timeout
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        ollama_response_json = response.json()

        # The actual JSON output is in the 'response' field of Ollama's response
        if 'response' in ollama_response_json and isinstance(ollama_response_json['response'], str):
            response_text = ollama_response_json['response'].strip()
            json_str_to_parse = None
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                json_str_to_parse = match.group(0)
                try:
                    extracted_data = json.loads(json_str_to_parse)
                    # Validate expected keys
                    if not all(key in extracted_data for key in
                               ["core_thesis", "key_concepts", "new_angle", "research_directions"]):
                        st.error(
                            f"Ollama Argument Extraction: Response missing some keys. Received: {extracted_data.keys()}")
                        return {**error_response_template, "core_thesis": "Error: Ollama response incomplete."}
                    return extracted_data
                except json.JSONDecodeError as e:
                    st.error(
                        f"Ollama Argument Extraction JSON Parsing Error: {e}. Attempted to parse: '{json_str_to_parse[:500]}'")
                    return {**error_response_template, "core_thesis": f"Error: Ollama JSON Parsing Error - {e}"}
            else:
                st.error(
                    f"Ollama Argument Extraction: No JSON object found in response. Response text: {response_text[:500]}")
                return {**error_response_template, "core_thesis": "Error: No JSON object in Ollama response."}
        else:
            st.error(
                f"Ollama Argument Extraction Error: 'response' field missing or not a string in API output. Full response: {ollama_response_json}")
            return {**error_response_template, "core_thesis": "Error: Ollama response format unexpected."}

    except requests.exceptions.RequestException as e:
        st.error(f"Ollama Argument Extraction API Request Error: {e}")
        return {**error_response_template, "core_thesis": f"Error: Ollama API Request Error - {e}"}
    except Exception as e:
        st.error(f"An unexpected error in Ollama Argument Extraction: {e}")
        return {**error_response_template, "core_thesis": f"Error: Unexpected error with Ollama - {e}"}


# Agent 1: Argument Extraction Agent
def argument_extraction_agent(gemini_model, base_paper_content: str, research_angle: str) -> Dict[str, str]:
    """Extract core thesis and identify new research direction with API key rotation or Ollama."""

    selected_model = st.session_state.get('selected_model', "Gemini")  # Default to Gemini if not set

    if selected_model == "Ollama":
        return argument_extraction_agent_ollama(base_paper_content, research_angle)

    # Existing Gemini Logic (ensure gemini_model is available)
    if not gemini_model:  # Added check for gemini_model
        st.error("Gemini model not available for argument extraction.")
        return {
            "core_thesis": "Error: Gemini model not available.",
            "key_concepts": [],
            "new_angle": research_angle,
            "research_directions": ["Error: Gemini model not available."]
        }

    if not st.session_state.get('gemini_api_keys_list') or not isinstance(st.session_state.gemini_api_keys_list,
                                                                          list) or not st.session_state.gemini_api_keys_list:
        st.error("Gemini API keys not configured or empty. Please set them in the sidebar.")
        return {
            "core_thesis": "Error: API keys not configured",
            "key_concepts": [],
            "new_angle": research_angle,
            "research_directions": ["Error: API keys not configured"]
        }

    prompt = f"""
    Analyze the following legal research paper and the student's new research angle.

    Base Paper Content:
    {base_paper_content[:3000]}  # Limit for API

    Student's Research Angle:
    {research_angle}

    Extract and provide a JSON response with these exact keys:
    - "core_thesis": The main argument of the base paper in 2-3 sentences
    - "key_concepts": A list of 3-5 key legal concepts discussed
    - "new_angle": The student's distinguishing research angle
    - "research_directions": A list of 2-3 potential research directions

    Example format:
    {{
        "core_thesis": "The paper argues that...",
        "key_concepts": ["concept1", "concept2", "concept3"],
        "new_angle": "Focus on...",
        "research_directions": ["Direction 1", "Direction 2"]
    }}
    """

    num_keys = len(st.session_state.gemini_api_keys_list)
    for attempt in range(num_keys):
        current_key_index = st.session_state.current_gemini_key_index
        current_api_key = st.session_state.gemini_api_keys_list[current_key_index]

        try:
            time.sleep(2)  # Added sleep before API call
            genai.configure(api_key=current_api_key, transport='rest')
            # Use the gemini_model instance passed, assuming genai.configure updates its underlying client.
            # If issues persist, one might need to re-initialize:
            # current_model_for_attempt = genai.GenerativeModel('gemini-2.0-flash')
            # response = current_model_for_attempt.generate_content(prompt)
            response = gemini_model.generate_content(prompt)

            response_text = response.text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)
            return result  # Success

        except google_exceptions.ResourceExhausted as e:
            st.warning(
                f"Rate limit hit with key ending: ...{current_api_key[-4:]} (Attempt {attempt + 1}/{num_keys}). Trying next key.")
            st.session_state.current_gemini_key_index = (current_key_index + 1) % num_keys
            if attempt == num_keys - 1:
                st.error("All Gemini API keys are currently rate-limited. Please try again later or add new keys.")
                # Fallback to original dynamic content if all keys fail
                angle_words = research_angle.split()[:10]
                content_words = base_paper_content.split()[:50]
                key_terms = [word for word in angle_words + content_words if
                             len(word) > 4 and word.lower() not in ['this', 'that', 'these', 'those', 'which', 'where',
                                                                    'when']]
                return {
                    "core_thesis": f"Analysis of legal aspects related to {' '.join(angle_words[:5])} (All keys rate-limited)",
                    "key_concepts": list(set(key_terms[:5])) if key_terms else ["legal analysis", "research",
                                                                                "regulation"],
                    "new_angle": research_angle,
                    "research_directions": [
                        f"Comparative analysis of {angle_words[0] if angle_words else 'topic'} (All keys rate-limited)",
                        f"Legal framework for {' '.join(angle_words[:3]) if angle_words else 'subject matter'} (All keys rate-limited)"
                    ]
                }
        except Exception as e:
            st.error(
                f"An unexpected error occurred with key ...{current_api_key[-4:]} (Attempt {attempt + 1}/{num_keys}): {e}")
            st.session_state.current_gemini_key_index = (
                                                                current_key_index + 1) % num_keys  # Rotate key on other errors too
            if attempt == num_keys - 1:
                # Fallback to original dynamic content if all keys fail
                angle_words = research_angle.split()[:10]
                content_words = base_paper_content.split()[:50]
                key_terms = [word for word in angle_words + content_words if
                             len(word) > 4 and word.lower() not in ['this', 'that', 'these', 'those', 'which', 'where',
                                                                    'when']]
                return {
                    "core_thesis": f"Analysis of legal aspects related to {' '.join(angle_words[:5])} (Error after trying all keys)",
                    "key_concepts": list(set(key_terms[:5])) if key_terms else ["legal analysis", "research",
                                                                                "regulation"],
                    "new_angle": research_angle,
                    "research_directions": [
                        f"Comparative analysis of {angle_words[0] if angle_words else 'topic'} (Error after trying all keys)",
                        f"Legal framework for {' '.join(angle_words[:3]) if angle_words else 'subject matter'} (Error after trying all keys)"
                    ]
                }

    # Fallback if loop somehow finishes without returning (should be caught by last attempt logic)
    angle_words = research_angle.split()[:10]
    content_words = base_paper_content.split()[:50]

    key_terms = [word for word in angle_words + content_words if
                 len(word) > 4 and word.lower() not in ['this', 'that', 'these', 'those', 'which', 'where', 'when']]
    return {
        "core_thesis": f"Analysis of legal aspects related to {' '.join(angle_words[:5])} (Failed to process with API)",
        "key_concepts": list(set(key_terms[:5])) if key_terms else ["legal analysis", "research", "regulation"],
        "new_angle": research_angle,
        "research_directions": [
            f"Comparative analysis of {angle_words[0] if angle_words else 'topic'} (Failed to process with API)",
            f"Legal framework for {' '.join(angle_words[:3]) if angle_words else 'subject matter'} (Failed to process with API)"
        ]
    }


# Agent 2: Keyword Generator Agent
def keyword_generator_agent(groq_client, extracted_args: Dict[str, str], seed_keywords: List[str] = []) -> List[str]:
    """Generate intelligent keyword permutations for search"""
    prompt = f"""
    Based on this legal research context, generate exactly 15 search keyword combinations.

    Core Thesis: {extracted_args.get('core_thesis', '')}
    Key Concepts: {', '.join(extracted_args.get('key_concepts', []))}
    New Research Angle: {extracted_args.get('new_angle', '')}
    Seed Keywords: {', '.join(seed_keywords)}

    Create varied keyword combinations for finding:
    - Supporting case law and precedents
    - Scholarly articles on the topic
    - Contrasting viewpoints
    - Recent developments

    Return ONLY a JSON array of strings, like:
    ["keyword combination 1", "keyword combination 2", ...]
    """

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=500
        )

        # Clean and parse response
        response_text = response.choices[0].message.content.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        keywords = json.loads(response_text)

        # Ensure it's a list
        if isinstance(keywords, list):
            return keywords
        else:
            raise ValueError("Response is not a list")

    except Exception as e:
        # Dynamic fallback based on extracted arguments
        base_keywords = []

        # Add angle-based keywords
        if extracted_args.get('new_angle'):
            angle_words = extracted_args['new_angle'].split()
            base_keywords.append(' '.join(angle_words[:5]))
            base_keywords.append(f"{' '.join(angle_words[:3])} legal")
            base_keywords.append(f"{' '.join(angle_words[:3])} case law")

        # Add concept-based keywords
        concepts = extracted_args.get('key_concepts', [])
        for i, concept in enumerate(concepts[:3]):
            base_keywords.append(f"{concept} legal framework")
            base_keywords.append(f"{concept} regulation")
            if i < len(concepts) - 1:
                base_keywords.append(f"{concept} {concepts[i + 1]}")

        # Add seed keywords
        if seed_keywords:
            base_keywords.extend(seed_keywords[:3])
            for seed in seed_keywords[:2]:
                base_keywords.append(f"{seed} legal precedent")

        # Generate topic-specific variations
        if concepts:
            base_keywords.extend([
                f"{concepts[0]} scholarly articles",
                f"{concepts[0]} recent developments",
                f"{concepts[0]} comparative analysis"
            ])

        # Ensure we have at least 15 keywords
        while len(base_keywords) < 15:
            if extracted_args.get('new_angle'):
                words = extracted_args['new_angle'].split()
                base_keywords.append(' '.join(words[i:i + 3]) for i in range(0, len(words) - 2, 2))
            else:
                break

        return base_keywords[:15]


# Agent 3: Source Crawler Agent
def source_crawler_agent(keywords: List[str], num_results: int = 5) -> List[Dict[str, Any]]:
    """Search multiple sources for relevant papers, cases, and news"""
    all_results = []

    # Tavily API search
    tavily_key = os.getenv("TAVILY_API_KEY")
    tavily_url = "https://api.tavily.com/search"

    # Serper API search
    serper_key = os.getenv("SERPER_API_KEY")
    serper_url = "https://google.serper.dev/search"

    # Search different types of sources
    source_types = [
        {"suffix": "scholarly article PDF", "type": "scholarly"},
        {"suffix": "case law judgment", "type": "case_law"},
        {"suffix": "legal news recent", "type": "news"}
    ]

    for keyword in keywords[:5]:  # Limit to avoid rate limits
        for source_type in source_types:
            # Tavily search with specific source type
            if source_type["type"] == "scholarly":
                tavily_query = f'site:ssrn.com OR site:jstor.org "{keyword}" filetype:pdf'
            elif source_type["type"] == "case_law":
                tavily_query = f'"{keyword}" judgment court case decision'
            else:  # news
                tavily_query = f'"{keyword}" legal news regulation 2024 2025'

            try:
                tavily_response = requests.post(
                    tavily_url,
                    json={
                        "api_key": tavily_key,
                        "query": tavily_query,
                        "search_depth": "advanced",
                        "max_results": num_results,
                        "include_raw_content": True,
                        "include_domains": ["ssrn.com", "jstor.org", "courtlistener.com", "law.com", "reuters.com"] if
                        source_type["type"] != "news" else []
                    }
                )
                if tavily_response.status_code == 200:
                    results = tavily_response.json().get('results', [])
                    for r in results:
                        # Skip SSRN landing pages
                        if "subscribe to this fee journal" not in r.get('content', '').lower():
                            all_results.append({
                                "title": r.get('title', ''),
                                "url": r.get('url', ''),
                                "snippet": r.get('content', '')[:500],  # Limit snippet length
                                "raw_content": r.get('raw_content', '')[:1000] if r.get('raw_content') else '',
                                "source": "Tavily",
                                "source_type": source_type["type"],
                                "keyword_used": keyword
                            })
            except:
                pass

        # Serper search for additional results
        try:
            # Search for PDFs and full texts
            serper_response = requests.post(
                serper_url,
                json={"q": f'{keyword} filetype:pdf OR "full text" legal research'},
                headers={"X-API-KEY": serper_key}
            )
            if serper_response.status_code == 200:
                results = serper_response.json().get('organic', [])[:num_results]
                for r in results:
                    if "pdf" in r.get('link', '').lower() or "full" in r.get('title', '').lower():
                        all_results.append({
                            "title": r.get('title', ''),
                            "url": r.get('link', ''),
                            "snippet": r.get('snippet', ''),
                            "raw_content": "",
                            "source": "Serper",
                            "source_type": "scholarly",
                            "keyword_used": keyword
                        })

            # Search for case law
            serper_response = requests.post(
                serper_url,
                json={"q": f'{keyword} "v." case judgment court'},
                headers={"X-API-KEY": serper_key}
            )
            if serper_response.status_code == 200:
                results = serper_response.json().get('organic', [])[:3]
                for r in results:
                    all_results.append({
                        "title": r.get('title', ''),
                        "url": r.get('link', ''),
                        "snippet": r.get('snippet', ''),
                        "raw_content": "",
                        "source": "Serper",
                        "source_type": "case_law",
                        "keyword_used": keyword
                    })
        except:
            pass

        time.sleep(0.5)  # Rate limiting

    # Remove duplicates based on URL
    seen_urls = set()
    unique_results = []
    for result in all_results:
        if result['url'] not in seen_urls:
            seen_urls.add(result['url'])
            unique_results.append(result)

    return unique_results


# Agent 4: Citation Chainer Agent (Ollama)
def citation_chainer_agent_ollama(top_papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract and follow citation trails using Ollama."""
    ollama_api_url = "https://apaims2.0.vassarlabs.com/ollama1/api/generate"
    chained_citations = []

    for paper in top_papers[:3]:  # Limit for performance
        prompt = f"""
        Based on this paper information:
        Title: {paper['title']}
        Snippet: {paper['snippet']}

        Suggest 3-5 related papers or cases that would likely be cited or relevant.
        Include:
        1. Landmark cases in this area
        2. Seminal scholarly works
        3. Recent developments

        Format as JSON list of objects, each with keys: "title", "relevance_reason", "search_terms".
        Example:
        [
            {{"title": "Example Case 1", "relevance_reason": "Landmark decision on topic X.", "search_terms": ["Example Case 1", "topic X"]}},
            {{"title": "Seminal Paper Y", "relevance_reason": "Foundation work for concept Z.", "search_terms": ["Seminal Paper Y", "concept Z"]}}
        ]
        """
        payload = {
            "model": "gemma3:12b",
            "prompt": prompt,
            "stream": False
        }

        advanced_options_str = st.session_state.get('ollama_advanced_options_str', '{}')
        if advanced_options_str and advanced_options_str.strip() != '{}':
            try:
                parsed_options = json.loads(advanced_options_str)
                if isinstance(parsed_options, dict) and parsed_options:
                    payload['options'] = parsed_options
                elif parsed_options:
                    st.warning(
                        f"Ollama Citation Chainer: Invalid format for advanced options: Expected a non-empty JSON object. Options ignored. Input: {advanced_options_str}")
            except json.JSONDecodeError:
                st.warning(
                    f"Ollama Citation Chainer: Could not parse advanced options (invalid JSON): {advanced_options_str}. Proceeding without them.")

        error_placeholder = {
            "title": f"Could not fetch citations for {paper.get('title', 'N/A')} via Ollama",
            "relevance_reason": "Ollama API error or parsing issue.",
            "search_terms": [],
            "parent_paper": paper.get('title', 'N/A')
        }

        try:
            response = requests.post(ollama_api_url, json=payload, timeout=60)
            response.raise_for_status()
            ollama_response_json = response.json()

            processed_successfully_for_paper = False
            if 'response' in ollama_response_json and isinstance(ollama_response_json['response'], str):
                response_text_content = ollama_response_json['response'].strip()
                json_str_to_parse = None
                citations_list = None

                match = re.search(r'\[[\s\S]*\]', response_text_content)
                if match:
                    json_str_to_parse = match.group(0)
                    try:
                        citations_list = json.loads(json_str_to_parse)
                        if isinstance(citations_list, list):
                            for citation in citations_list:
                                if isinstance(citation,
                                              dict) and "title" in citation and "relevance_reason" in citation and "search_terms" in citation:
                                    citation['parent_paper'] = paper['title']
                                    chained_citations.append(citation)
                                    processed_successfully_for_paper = True
                                else:
                                    st.warning(
                                        f"Ollama Citation Chainer: Skipping malformed citation object: {citation} for paper {paper.get('title')}")
                            if not processed_successfully_for_paper and not citations_list:  # List was empty
                                st.warning(
                                    f"Ollama Citation Chainer: Extracted JSON list was empty for paper {paper.get('title')}")
                            elif not processed_successfully_for_paper:  # List had items, but all were malformed
                                st.warning(
                                    f"Ollama Citation Chainer: All citation objects were malformed in list for paper {paper.get('title')}")
                        else:
                            st.error(
                                f"Ollama Citation Chainer: Regex found a list, but parsing resulted in type {type(citations_list)} for paper {paper.get('title')}")
                    except json.JSONDecodeError as json_err:
                        st.error(
                            f"Ollama Citation Chainer JSONDecodeError: {json_err} for paper '{paper.get('title', 'N/A')}'. Attempted to parse from regex: '{json_str_to_parse[:500]}'")
                else:  # Regex did not find a list
                    st.warning(
                        f"Ollama Citation Chainer: No JSON list found in response for paper '{paper.get('title', 'N/A')}'. Response: {response_text_content[:200]}")
            else:  # 'response' field missing or not a string
                st.error(
                    f"Ollama Citation Chainer Error: 'response' field missing or invalid. Full response: {ollama_response_json} for paper {paper.get('title')}")

        except requests.exceptions.RequestException as e:
            st.error(f"Ollama Citation Chainer API Request Error: {e} for paper {paper.get('title')}")
        except Exception as e:
            st.error(f"Unexpected error in Ollama Citation Chainer for paper {paper.get('title', 'N/A')}: {e}")

        if not processed_successfully_for_paper:
            chained_citations.append(error_placeholder)

    return chained_citations


# Agent 4: Citation Chainer Agent
def citation_chainer_agent(gemini_model, top_papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract and follow citation trails with API key rotation or Ollama."""

    selected_model = st.session_state.get('selected_model', "Gemini")

    if selected_model == "Ollama":
        return citation_chainer_agent_ollama(top_papers)

    # Existing Gemini Logic
    if not gemini_model:
        st.error("Citation Chainer: Gemini model not available.")
        return [{"title": "Error: Gemini model not available.", "relevance_reason": "", "search_terms": [],
                 "parent_paper": "N/A"}]

    if not st.session_state.get('gemini_api_keys_list') or not isinstance(st.session_state.gemini_api_keys_list,
                                                                          list) or not st.session_state.gemini_api_keys_list:
        st.error("Citation Chainer: Gemini API keys not configured or empty.")
        return [{"title": "Error: API keys not configured.", "relevance_reason": "", "search_terms": [],
                 "parent_paper": "N/A"}]

    chained_citations = []
    num_keys = len(st.session_state.gemini_api_keys_list)

    for paper in top_papers[:3]:  # Limit for performance
        prompt = f"""
        Based on this paper information:
        Title: {paper['title']}
        Snippet: {paper['snippet']}

        Suggest 3-5 related papers or cases that would likely be cited or relevant.
        Include:
        1. Landmark cases in this area
        2. Seminal scholarly works
        3. Recent developments

        Format as JSON list with keys: title, relevance_reason, search_terms
        """

        processed_successfully = False
        for attempt in range(num_keys):
            current_key_index = st.session_state.current_gemini_key_index
            current_api_key = st.session_state.gemini_api_keys_list[current_key_index]

            try:
                time.sleep(2)  # Added sleep before API call
                genai.configure(api_key=current_api_key, transport='rest')
                # As before, assuming genai.configure updates the existing model instance.
                # If not, re-initialize: current_model_for_attempt = genai.GenerativeModel('gemini-2.0-flash')
                response = gemini_model.generate_content(prompt)

                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif response_text.startswith("```"):  # Handle cases where only ``` is present
                    response_text = response_text.split("```")[1].split("```")[0].strip()

                citations = json.loads(response_text)
                for citation in citations:
                    citation['parent_paper'] = paper['title']
                    chained_citations.append(citation)
                processed_successfully = True
                break  # Success for this paper, move to next paper

            except google_exceptions.ResourceExhausted as e:
                st.warning(
                    f"Rate limit hit for Citation Chainer with key ...{current_api_key[-4:]} (Paper: {paper.get('title', 'N/A')}, Attempt {attempt + 1}/{num_keys}). Trying next key.")
                st.session_state.current_gemini_key_index = (current_key_index + 1) % num_keys
                if attempt == num_keys - 1:
                    st.error(f"All keys rate-limited while processing citations for paper: {paper.get('title', 'N/A')}")
            except Exception as e:
                st.error(
                    f"An error in Citation Chainer with key ...{current_api_key[-4:]} for paper {paper.get('title', 'N/A')} (Attempt {attempt + 1}/{num_keys}): {e}")
                st.session_state.current_gemini_key_index = (current_key_index + 1) % num_keys
                if attempt == num_keys - 1:
                    st.error(
                        f"Failed to process citations for paper {paper.get('title', 'N/A')} after trying all keys.")

        if not processed_successfully:
            # Add a placeholder or note if this paper couldn't be processed
            chained_citations.append({
                "title": f"Could not fetch citations for {paper.get('title', 'N/A')}",
                "relevance_reason": "API error or all keys rate-limited.",
                "search_terms": [],
                "parent_paper": paper.get('title', 'N/A')
            })

    return chained_citations


# Agent 5: Relevance Scorer Agent
def relevance_scorer_agent(groq_client, papers: List[Dict[str, Any]], research_context: Dict[str, str]) -> List[
    Dict[str, Any]]:
    """Score and rank papers by relevance"""
    scored_papers = []

    # Extract key terms from research context for dynamic scoring
    angle_terms = research_context.get('new_angle', '').lower().split()
    concept_terms = [c.lower() for c in research_context.get('key_concepts', [])]
    all_context_terms = angle_terms + concept_terms

    limit = st.session_state.get('max_sources_to_analyze', 20)  # Default to 20 for relevance scoring
    papers_to_score = papers if limit == 0 else papers[:limit]

    for paper in papers_to_score:
        prompt = f"""
        Score this paper's relevance (0-100) for the research context:

        Research Angle: {research_context.get('new_angle', '')}
        Core Concepts: {', '.join(research_context.get('key_concepts', []))}

        Paper:
        Title: {paper['title']}
        Snippet: {paper['snippet']}

        Return a JSON object with exactly these keys:
        - "score": integer between 0-100
        - "reason": one sentence explaining the relevance
        - "insights": key takeaways in 1-2 sentences

        Example format:
        {{"score": 85, "reason": "Directly addresses the research topic", "insights": "Provides framework for analysis"}}
        """

        try:
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=300
            )

            # Clean and parse the response
            response_text = response.choices[0].message.content.strip()
            # Remove any markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            scoring = json.loads(response_text)
            paper['relevance_score'] = int(scoring.get('score', 50))
            paper['relevance_reason'] = scoring.get('reason', 'Relevance determined by title and content match')
            paper['key_insights'] = scoring.get('insights', 'Relevant to research topic')
            scored_papers.append(paper)
        except Exception as e:
            # Dynamic fallback scoring based on actual research context
            title_lower = paper.get('title', '').lower()
            snippet_lower = paper.get('snippet', '').lower()

            # Dynamic keyword-based scoring
            score = 40  # Base score
            reasons = []

            # Check for research angle terms
            angle_matches = sum(1 for term in angle_terms if term in title_lower or term in snippet_lower)
            if angle_matches > 0:
                score += min(angle_matches * 10, 30)
                reasons.append(f"matches research angle ({angle_matches} terms)")

            # Check for concept matches
            concept_matches = sum(1 for term in concept_terms if term in title_lower or term in snippet_lower)
            if concept_matches > 0:
                score += min(concept_matches * 8, 24)
                reasons.append(f"contains key concepts ({concept_matches} found)")

            # Check for legal/regulation mentions
            if any(word in title_lower for word in ['regulat', 'govern', 'policy', 'law', 'legal']):
                score += 10
                reasons.append("legal/regulatory focus")

            # Check for case law indicators
            if ' v. ' in title_lower or 'case' in title_lower:
                score += 8
                reasons.append("case law")

            # Source type bonus
            if paper.get('source_type') == 'scholarly':
                score += 5
            elif paper.get('source_type') == 'case_law':
                score += 7

            paper['relevance_score'] = min(score, 95)  # Cap at 95
            paper['relevance_reason'] = f"Relevance: {', '.join(reasons) if reasons else 'general topical match'}"
            paper['key_insights'] = f"May provide insights on {' and '.join(all_context_terms[:3])}"
            scored_papers.append(paper)

    # Sort by relevance score
    scored_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    return scored_papers


# Agent: Summary Extraction Agent (Ollama)
def summary_extraction_agent_ollama(papers: List[Dict[str, Any]], research_context: Dict[str, str]) -> List[
    Dict[str, Any]]:
    """Extract meaningful summaries from paper content using Ollama."""
    ollama_api_url = "https://apaims2.0.vassarlabs.com/ollama1/api/generate"
    research_topic = research_context.get('new_angle', 'legal research')

    # Limit is applied by the caller (summary_extraction_agent)
    # So, summary_extraction_agent_ollama processes all papers passed to it.

    for paper in papers:
        if not paper.get('raw_content') or len(paper.get('raw_content', '')) <= 100:
            paper['extracted_summary'] = paper.get('snippet', paper.get('title', 'No content for summary.'))[:200]
            paper['key_findings'] = paper.get('key_findings', ["Snippet used as summary due to short content."])
            paper['topic_relevance'] = paper.get('topic_relevance', f"Relevant to {research_topic} (based on snippet)")
            continue

        prompt = f"""
        Extract a concise summary from this paper content related to: {research_topic}

        Title: {paper['title']}
        Content: {paper['raw_content'][:st.session_state.get('ollama_max_prompt_chars', 1500)]} # Max content length

        Provide JSON output with these exact keys:
        - "main_argument": Main argument/thesis (1-2 sentences)
        - "key_findings": Key findings or principles (2-3 bullet points as a list of strings)
        - "topic_relevance": Relevance to the research topic: {research_topic} (1 sentence)

        Example:
        {{
            "main_argument": "The paper argues X and Y.",
            "key_findings": ["Finding A.", "Finding B."],
            "topic_relevance": "This paper directly addresses Z."
        }}
        """
        payload = {
            "model": "gemma3:12b",
            "prompt": prompt,
            "stream": False
        }

        advanced_options_str = st.session_state.get('ollama_advanced_options_str', '{}')
        if advanced_options_str and advanced_options_str.strip() != '{}':
            try:
                parsed_options = json.loads(advanced_options_str)
                if isinstance(parsed_options, dict) and parsed_options:
                    payload['options'] = parsed_options
                elif parsed_options:
                    st.warning(
                        f"Ollama Summary Extraction: Invalid format for advanced options: Expected a non-empty JSON object. Options ignored. Input: {advanced_options_str}")
            except json.JSONDecodeError:
                st.warning(
                    f"Ollama Summary Extraction: Could not parse advanced options (invalid JSON): {advanced_options_str}. Proceeding without them.")

        error_summary = paper.get('snippet', paper.get('title', 'Error: No summary available via Ollama'))[
                        :200] + " (Ollama Error)"
        error_findings = ["Ollama Error"]
        error_relevance = "Ollama Error"

        try:
            response = requests.post(ollama_api_url, json=payload, timeout=60)
            response.raise_for_status()
            ollama_response_json = response.json()

            if 'response' in ollama_response_json and isinstance(ollama_response_json['response'], str):
                response_text = ollama_response_json['response'].strip()
                json_str_to_parse = None
                match = re.search(r'\{[\s\S]*\}', response_text)
                if match:
                    json_str_to_parse = match.group(0)
                    try:
                        summary_data = json.loads(json_str_to_parse)
                        paper['extracted_summary'] = summary_data.get('main_argument', error_summary)
                        paper['key_findings'] = summary_data.get('key_findings', error_findings)
                        paper['topic_relevance'] = summary_data.get('topic_relevance', error_relevance)
                        if not isinstance(paper['key_findings'], list):
                            paper['key_findings'] = [str(paper['key_findings'])] if paper[
                                'key_findings'] else error_findings
                    except json.JSONDecodeError as e:
                        st.error(
                            f"Ollama Summary Extraction JSON Parsing Error: {e}. Attempted to parse: '{json_str_to_parse[:500]}' for paper {paper.get('title')}")
                        paper['extracted_summary'], paper['key_findings'], paper[
                            'topic_relevance'] = error_summary, error_findings, error_relevance
                else:
                    st.error(
                        f"Ollama Summary Extraction: No JSON object found in response for paper {paper.get('title')}. Response: {response_text[:500]}")
                    paper['extracted_summary'], paper['key_findings'], paper[
                        'topic_relevance'] = error_summary, error_findings, error_relevance
            else:
                st.error(
                    f"Ollama Summary Error: 'response' field missing or invalid. Full response: {ollama_response_json} for paper {paper.get('title')}")
                paper['extracted_summary'], paper['key_findings'], paper[
                    'topic_relevance'] = error_summary, error_findings, error_relevance
        except requests.exceptions.RequestException as e:
            st.error(f"Ollama Summary API Request Error: {e} for paper {paper.get('title')}")
            paper['extracted_summary'], paper['key_findings'], paper[
                'topic_relevance'] = error_summary, error_findings, error_relevance
        except Exception as e:
            st.error(f"Unexpected error in Ollama Summary Extraction: {e} for paper {paper.get('title')}")
            paper['extracted_summary'], paper['key_findings'], paper[
                'topic_relevance'] = error_summary, error_findings, error_relevance

    return papers


# Add new Summary Extraction Agent after the citation chainer agent
def summary_extraction_agent(gemini_model, papers: List[Dict[str, Any]], research_context: Dict[str, str]) -> List[
    Dict[str, Any]]:
    """Extract meaningful summaries from paper content with API key rotation or Ollama."""

    selected_model = st.session_state.get('selected_model', "Gemini")

    if selected_model == "Ollama":
        return summary_extraction_agent_ollama(papers, research_context)

    # Existing Gemini Logic
    if not gemini_model:
        st.error("Summary Extraction: Gemini model not available.")
        for paper in papers:
            paper['extracted_summary'] = paper.get('snippet', paper.get('title',
                                                                        'Error: Gemini model not available')) + " (Gemini Model Error)"
            paper['key_findings'] = ["Gemini Model Error"]
            paper['topic_relevance'] = "Gemini Model Error"
        return papers

    if not st.session_state.get('gemini_api_keys_list') or not isinstance(st.session_state.gemini_api_keys_list,
                                                                          list) or not st.session_state.gemini_api_keys_list:
        st.error("Summary Extraction: Gemini API keys not configured or empty.")
        for paper in papers:
            paper['extracted_summary'] = paper.get('snippet', paper.get('title',
                                                                        'Error: No summary available')) + " (API Key Error)"
            paper['key_findings'] = ["API Key Error"]
            paper['topic_relevance'] = "API Key Error"
        return papers

    research_topic = research_context.get('new_angle', 'legal research')
    num_keys = len(st.session_state.gemini_api_keys_list)

    # Determine the list of papers to process based on the session state limit
    limit = st.session_state.get('max_sources_to_analyze', 10)  # Default to 10 for summary extraction
    papers_to_process = papers if limit == 0 else papers[:limit]

    for paper in papers_to_process:
        if not paper.get('raw_content') or len(paper['raw_content']) <= 100:
            paper['extracted_summary'] = paper.get('snippet', paper.get('title', 'No content for summary.'))[:200]
            paper['key_findings'] = paper.get('key_findings', [])
            paper['topic_relevance'] = paper.get('topic_relevance', f"Relevant to {research_topic}")
            continue

        prompt = f"""
        Extract a concise summary from this paper content related to: {research_topic}

        Title: {paper['title']}
        Content: {paper['raw_content'][:1500]}

        Provide:
        1. Main argument/thesis (1-2 sentences)
        2. Key findings or principles (2-3 bullet points)
        3. Relevance to the research topic: {research_topic}

        Format as JSON with keys: main_argument, key_findings, topic_relevance
        """

        processed_successfully = False
        for attempt in range(num_keys):
            current_key_index = st.session_state.current_gemini_key_index
            current_api_key = st.session_state.gemini_api_keys_list[current_key_index]

            try:
                time.sleep(2)  # Added sleep before API call
                genai.configure(api_key=current_api_key, transport='rest')
                # current_model_for_attempt = genai.GenerativeModel('gemini-2.0-flash')
                # response = current_model_for_attempt.generate_content(prompt)
                response = gemini_model.generate_content(prompt)

                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif response_text.startswith("```"):
                    response_text = response_text.split("```")[1].split("```")[0].strip()

                summary_data = json.loads(response_text)
                paper['extracted_summary'] = summary_data.get('main_argument',
                                                              paper.get('snippet', 'Error parsing summary')[:200])
                paper['key_findings'] = summary_data.get('key_findings', [])
                paper['topic_relevance'] = summary_data.get('topic_relevance',
                                                            f"Relevance to {research_topic} unclear after parsing error.")
                processed_successfully = True
                break  # Success for this paper

            except google_exceptions.ResourceExhausted as e:
                st.warning(
                    f"Rate limit hit for Summary Extraction with key ...{current_api_key[-4:]} (Paper: {paper.get('title', 'N/A')}, Attempt {attempt + 1}/{num_keys}). Trying next key.")
                st.session_state.current_gemini_key_index = (current_key_index + 1) % num_keys
                if attempt == num_keys - 1:
                    st.error(f"All keys rate-limited while extracting summary for paper: {paper.get('title', 'N/A')}")
            except Exception as e:
                st.error(
                    f"An error in Summary Extraction with key ...{current_api_key[-4:]} for paper {paper.get('title', 'N/A')} (Attempt {attempt + 1}/{num_keys}): {e}")
                st.session_state.current_gemini_key_index = (current_key_index + 1) % num_keys
                if attempt == num_keys - 1:
                    st.error(f"Failed to extract summary for paper {paper.get('title', 'N/A')} after trying all keys.")

        if not processed_successfully:
            paper['extracted_summary'] = paper.get('snippet', paper.get('title', 'Error: No summary available'))[
                                         :200] + " (API Error or All Keys Rate-Limited)"
            paper['key_findings'] = ["API Error or All Keys Rate-Limited"]
            paper['topic_relevance'] = "API Error or All Keys Rate-Limited"

    return papers


# Agent for Case Analysis (Ollama)
def case_analysis_agent_ollama(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyzes legal case documents from a list of papers to extract structured information using Ollama."""
    ollama_api_url = "https://apaims2.0.vassarlabs.com/ollama1/api/generate"

    default_ollama_error_values = {
        'case_facts': "Ollama Error: Could not analyze.",
        'legal_issues': ["Ollama Error"],
        'arguments': {"plaintiff": "Ollama Error", "defendant": "Ollama Error"},
        'judgment': "Ollama Error",
        'court_findings': "Ollama Error"
    }

    analysis_limit = st.session_state.get('max_sources_to_analyze', 0)  # 0 means all case law papers
    processed_case_law_count = 0

    for paper in papers:
        if paper.get('source_type') == 'case_law':
            if analysis_limit != 0 and processed_case_law_count >= analysis_limit:
                # Add placeholder if limit is reached and this paper won't be processed
                paper.update({k: paper.get(k, "Not analyzed due to limit.") for k in
                              ['case_facts', 'legal_issues', 'arguments', 'judgment', 'court_findings']})
                continue

            content_to_analyze = paper.get('raw_content', '') if len(paper.get('raw_content', '')) > 200 else paper.get(
                'snippet', '')
            if not content_to_analyze: content_to_analyze = paper.get('title', '')
            truncated_content = content_to_analyze[
                                :st.session_state.get('ollama_max_prompt_chars', 2000)]  # Ollama context limit

            prompt = f"""
            Analyze the following legal case based on its title and content.
            Provide a JSON response with these exact keys: "case_facts", "legal_issues" (as a list of strings), "arguments" (as a dict with "plaintiff" and "defendant" keys), "judgment", "court_findings".

            Title: {paper.get('title', 'N/A')}
            Content: {truncated_content}

            If specific details are not found, use "Not available" or an empty list/dictionary as appropriate for the field type.
            Example JSON format:
            {{
                "case_facts": "Summary of facts...",
                "legal_issues": ["Issue 1", "Issue 2"],
                "arguments": {{
                    "plaintiff": "Plaintiff's arguments...",
                    "defendant": "Defendant's arguments..."
                }},
                "judgment": "The court decided...",
                "court_findings": "The court found that..."
            }}
            """
            payload = {
                "model": "gemma3:12b",
                "prompt": prompt,
                "stream": False
            }

        advanced_options_str = st.session_state.get('ollama_advanced_options_str', '{}')
        if advanced_options_str and advanced_options_str.strip() != '{}':
            try:
                parsed_options = json.loads(advanced_options_str)
                if isinstance(parsed_options, dict) and parsed_options:
                    payload['options'] = parsed_options
                elif parsed_options:
                    st.warning(
                        f"Ollama Case Analysis: Invalid format for advanced options: Expected a non-empty JSON object. Options ignored. Input: {advanced_options_str}")
            except json.JSONDecodeError:
                st.warning(
                    f"Ollama Case Analysis: Could not parse advanced options (invalid JSON): {advanced_options_str}. Proceeding without them.")

            try:
                response = requests.post(ollama_api_url, json=payload, timeout=60)
                response.raise_for_status()
                ollama_response_json = response.json()

                if 'response' in ollama_response_json and isinstance(ollama_response_json['response'], str):
                    response_text = ollama_response_json['response'].strip()
                    json_str_to_parse = None
                    match = re.search(r'\{[\s\S]*\}', response_text)
                    if match:
                        json_str_to_parse = match.group(0)
                        try:
                            analysis_results = json.loads(json_str_to_parse)
                            paper['case_facts'] = analysis_results.get('case_facts',
                                                                       default_ollama_error_values['case_facts'])
                            paper['legal_issues'] = analysis_results.get('legal_issues',
                                                                         default_ollama_error_values['legal_issues'])
                            paper['arguments'] = analysis_results.get('arguments',
                                                                      default_ollama_error_values['arguments'])
                            paper['judgment'] = analysis_results.get('judgment',
                                                                     default_ollama_error_values['judgment'])
                            paper['court_findings'] = analysis_results.get('court_findings',
                                                                           default_ollama_error_values[
                                                                               'court_findings'])

                            if not isinstance(paper['arguments'], dict) or not all(
                                    k in paper['arguments'] for k in ['plaintiff', 'defendant']):
                                paper['arguments'] = default_ollama_error_values['arguments']
                            if not isinstance(paper['legal_issues'], list):
                                paper['legal_issues'] = [str(paper['legal_issues'])] if paper['legal_issues'] else \
                                    default_ollama_error_values['legal_issues']
                        except json.JSONDecodeError as e:
                            st.error(
                                f"Ollama Case Analysis JSON Parsing Error: {e}. Attempted to parse: '{json_str_to_parse[:500]}' for paper {paper.get('title')}")
                            paper.update(default_ollama_error_values)
                    else:
                        st.error(
                            f"Ollama Case Analysis: No JSON object found in response for paper {paper.get('title')}. Response: {response_text[:500]}")
                        paper.update(default_ollama_error_values)
                else:
                    st.error(
                        f"Ollama Case Analysis Error: 'response' field missing or invalid. Response: {ollama_response_json} for paper {paper.get('title')}")
                    paper.update(default_ollama_error_values)
            except requests.exceptions.RequestException as e:
                st.error(f"Ollama Case Analysis API Request Error: {e} for paper {paper.get('title')}")
                paper.update(default_ollama_error_values)
            except Exception as e:
                st.error(f"Unexpected error in Ollama Case Analysis: {e} for paper {paper.get('title')}")
                paper.update(default_ollama_error_values)

            processed_case_law_count += 1  # Increment after processing attempt
        else:
            # Ensure non-case_law papers also have these fields if expected by UI, though typically not accessed.
            paper.update({k: paper.get(k, "N/A - Not a case") for k in
                          ['case_facts', 'legal_issues', 'arguments', 'judgment', 'court_findings']})

    return papers


# Agent for Case Analysis
def case_analysis_agent(gemini_model, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyzes legal case documents from a list of papers to extract structured information using either Gemini (with API key rotation) or Ollama.
    """
    selected_model = st.session_state.get('selected_model', "Gemini")

    if selected_model == "Ollama":
        return case_analysis_agent_ollama(papers)

    # Existing Gemini Logic
    if not gemini_model:
        st.error("Case Analysis: Gemini model not available.")
        for paper in papers:
            if paper.get('source_type') == 'case_law':
                paper.update({
                    'case_facts': "Error: Gemini model not available.",
                    'legal_issues': ["Gemini model not available"],
                    'arguments': {"plaintiff": "Gemini model not available", "defendant": "Gemini model not available"},
                    'judgment': "Gemini model not available",
                    'court_findings': "Gemini model not available"
                })
        return papers

    if not st.session_state.get('gemini_api_keys_list') or not isinstance(st.session_state.gemini_api_keys_list,
                                                                          list) or not st.session_state.gemini_api_keys_list:
        st.error("Case Analysis: Gemini API keys not configured or empty.")
        for paper in papers:
            if paper.get('source_type') == 'case_law':
                paper.update({
                    'case_facts': "Error: API keys not configured",
                    'legal_issues': ["API keys not configured"],
                    'arguments': {"plaintiff": "API keys not configured", "defendant": "API keys not configured"},
                    'judgment': "API keys not configured",
                    'court_findings': "API keys not configured"
                })
        return papers

    num_keys = len(st.session_state.gemini_api_keys_list)

    analysis_limit = st.session_state.get('max_sources_to_analyze', 0)  # 0 means all case law papers
    processed_case_law_count = 0

    for paper in papers:
        if paper.get('source_type') == 'case_law':
            if analysis_limit != 0 and processed_case_law_count >= analysis_limit:
                # Add placeholder if limit is reached and this paper won't be processed
                paper.update({
                    'case_facts': "Not analyzed due to limit.",
                    'legal_issues': ["Not analyzed due to limit."],
                    'arguments': {"plaintiff": "Not analyzed due to limit.", "defendant": "Not analyzed due to limit."},
                    'judgment': "Not analyzed due to limit.",
                    'court_findings': "Not analyzed due to limit."
                })
                continue

            default_error_values = {
                'case_facts': "Failed to analyze after trying all keys.",
                'legal_issues': ["Failed to analyze after trying all keys."],
                'arguments': {"plaintiff": "Failed to analyze", "defendant": "Failed to analyze"},
                'judgment': "Failed to analyze after trying all keys.",
                'court_findings': "Failed to analyze after trying all keys."
            }

            content_to_analyze = paper.get('raw_content', '') if len(paper.get('raw_content', '')) > 200 else paper.get(
                'snippet', '')
            if not content_to_analyze: content_to_analyze = paper.get('title', '')
            truncated_content = content_to_analyze[:2000]

            prompt = f"""
            Analyze the following legal case based on its title and content.
            Provide a JSON response with these exact keys: "case_facts", "legal_issues", "arguments" (as a dict with "plaintiff" and "defendant"), "judgment", "court_findings".

            Title: {paper.get('title', 'N/A')}
            Content: {truncated_content}

            If specific details are not found, use "Not available" or an empty list/dictionary as appropriate for the field type.
            Example JSON format:
            {{
                "case_facts": "Summary of facts...",
                "legal_issues": ["Issue 1", "Issue 2"],
                "arguments": {{
                    "plaintiff": "Plaintiff's arguments...",
                    "defendant": "Defendant's arguments..."
                }},
                "judgment": "The court decided...",
                "court_findings": "The court found that..."
            }}
            """

            processed_successfully = False
            for attempt in range(num_keys):
                current_key_index = st.session_state.current_gemini_key_index
                current_api_key = st.session_state.gemini_api_keys_list[current_key_index]

                try:
                    time.sleep(2)  # Added sleep before API call
                    genai.configure(api_key=current_api_key, transport='rest')
                    response = gemini_model.generate_content(prompt)

                    response_text = response.text.strip()
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]

                    analysis_results = json.loads(response_text)

                    paper['case_facts'] = analysis_results.get('case_facts', "Not available")
                    paper['legal_issues'] = analysis_results.get('legal_issues', [])
                    paper['arguments'] = analysis_results.get('arguments', {"plaintiff": "Not available",
                                                                            "defendant": "Not available"})
                    # Ensure 'arguments' is a dict with 'plaintiff' and 'defendant' keys
                    if not isinstance(paper['arguments'], dict) or \
                            not ('plaintiff' in paper['arguments'] and 'defendant' in paper['arguments']):
                        paper['arguments'] = {"plaintiff": "Not available", "defendant": "Not available"}
                    paper['judgment'] = analysis_results.get('judgment', "Not available")
                    paper['court_findings'] = analysis_results.get('court_findings', "Not available")
                    processed_successfully = True
                    break

                except google_exceptions.ResourceExhausted as e:
                    st.warning(
                        f"Rate limit hit for Case Analysis with key ...{current_api_key[-4:]} (Paper: {paper.get('title', 'N/A')}, Attempt {attempt + 1}/{num_keys}). Trying next key.")
                    st.session_state.current_gemini_key_index = (current_key_index + 1) % num_keys
                    if attempt == num_keys - 1:
                        st.error(f"All keys rate-limited while analyzing case: {paper.get('title', 'N/A')}")
                        paper.update(default_error_values)
                except json.JSONDecodeError as e:
                    st.warning(
                        f"Case Analysis: Error decoding JSON for paper '{paper.get('title', 'Unknown Title')}' with key ...{current_api_key[-4:]} (Attempt {attempt + 1}/{num_keys}): {e}. Raw: {response_text[:100]}")
                    st.session_state.current_gemini_key_index = (current_key_index + 1) % num_keys
                    if attempt == num_keys - 1:
                        st.error(
                            f"Failed to decode case analysis for {paper.get('title', 'N/A')} after trying all keys.")
                        paper.update(default_error_values)
                except Exception as e:
                    st.error(
                        f"An error in Case Analysis with key ...{current_api_key[-4:]} for paper {paper.get('title', 'N/A')} (Attempt {attempt + 1}/{num_keys}): {e}")
                    st.session_state.current_gemini_key_index = (current_key_index + 1) % num_keys
                    if attempt == num_keys - 1:
                        st.error(f"Failed to analyze case {paper.get('title', 'N/A')} after trying all keys.")
                        paper.update(default_error_values)

            if not processed_successfully:
                paper.update(default_error_values)

            processed_case_law_count += 1  # Increment after processing attempt (successful or not)

    return papers


def format_citation(paper: Dict[str, Any], style: str = "APA") -> str:
    """Format citation in requested style"""
    title = paper.get('title', 'Unknown Title')
    url = paper.get('url', '')
    date = datetime.now().strftime("%Y, %B %d")

    if style == "APA":
        return f"{title}. Retrieved {date}, from {url}"
    elif style == "Bluebook":
        return f"{title}, available at {url} (last visited {date})"
    else:
        return f"{title}. {url}"


# Main Streamlit App
def main():
    st.title("⚖️ Multi-Agent Legal Research Companion")
    st.markdown("### AI-Powered Research Assistant for Law Students")

    # Initialize session state variables if they don't exist
    # if 'gemini_api_tokens_str' not in st.session_state: # Removed as keys come from .env
    #     st.session_state.gemini_api_tokens_str = ""
    if 'gemini_api_keys_list' not in st.session_state:
        st.session_state.gemini_api_keys_list = []
    if 'current_gemini_key_index' not in st.session_state:
        st.session_state.current_gemini_key_index = 0
    if 'clients_initialized' not in st.session_state:
        st.session_state.clients_initialized = False
    if 'gemini_model' not in st.session_state:
        st.session_state.gemini_model = None
    if 'groq_client' not in st.session_state:  # Initialize Groq client placeholder
        st.session_state.groq_client = None
    if 'ollama_embeddings' not in st.session_state:
        st.session_state.ollama_embeddings = None
    if 'chroma_client' not in st.session_state:
        st.session_state.chroma_client = None
    if 'chroma_collection' not in st.session_state:
        st.session_state.chroma_collection = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Gemini"  # Default to Gemini
    if 'max_sources_to_analyze' not in st.session_state:
        st.session_state.max_sources_to_analyze = 10  # Default value
    if 'ollama_max_prompt_chars' not in st.session_state:
        st.session_state.ollama_max_prompt_chars = 2500  # Default value (e.g., 2500 characters)
    if 'ollama_advanced_options_str' not in st.session_state:
        st.session_state.ollama_advanced_options_str = "{}"  # Default to an empty JSON object string
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # List of tuples (query, response) or dicts

    # Load API Keys from .env and initialize clients
    gemini_api_keys_env = os.getenv("GEMINI_API_KEYS")
    if gemini_api_keys_env:
        st.session_state.gemini_api_keys_list = [key.strip() for key in gemini_api_keys_env.split(',') if key.strip()]
        if st.session_state.gemini_api_keys_list:
            st.session_state.current_gemini_key_index = 0
            first_gemini_key = st.session_state.gemini_api_keys_list[0]
            
            # Initialize Gemini and Groq clients
            # init_clients will use the first Gemini key and GROQ_API_KEY from .env
            st.session_state.gemini_model, st.session_state.groq_client = init_clients(api_key=first_gemini_key)
            
            if st.session_state.gemini_model:
                st.session_state.clients_initialized = True # This flag might still be useful
                st.sidebar.success(f"Gemini initialized using the first of {len(st.session_state.gemini_api_keys_list)} key(s) from .env.")
                # Initialize Ollama embeddings if Gemini is up
                if not st.session_state.ollama_embeddings:
                    st.session_state.ollama_embeddings = init_ollama_embeddings(OLLAMA_EMBEDDING_BASE_URL, EMBEDDING_MODEL)
            else:
                st.session_state.clients_initialized = False
                st.sidebar.error("Failed to initialize Gemini with the first key from .env. Check GEMINI_API_KEYS.")
            
            # Check Groq client status (it's initialized within init_clients)
            if st.session_state.groq_client:
                st.sidebar.info("Groq client also attempted initialization via init_clients.")
            elif not os.getenv("GROQ_API_KEY"):
                st.sidebar.warning("GROQ_API_KEY not in .env. Groq features might be limited.")
            else: # GROQ_API_KEY was present but client is None
                st.sidebar.error("Groq client failed to initialize despite key in .env.")

        else: # No valid keys after parsing GEMINI_API_KEYS
            st.session_state.clients_initialized = False
            st.sidebar.error("No valid Gemini API tokens found in GEMINI_API_KEYS from .env.")
            # Attempt to init Groq if Gemini keys are missing/invalid but Groq key might be present
            if not st.session_state.groq_client and os.getenv("GROQ_API_KEY"):
                _, st.session_state.groq_client = init_clients(api_key="DUMMY_GEMINI_KEY_FOR_GROQ_INIT") # Pass a dummy, init_clients handles Groq separately
                if st.session_state.groq_client:
                     st.sidebar.info("Groq client initialized independently as Gemini keys were an issue.")
                else:
                     st.sidebar.error("Groq client also failed to initialize.")
    else: # GEMINI_API_KEYS not in .env
        st.session_state.clients_initialized = False
        st.sidebar.error("GEMINI_API_KEYS not found in .env. Gemini features unavailable.")
        # Attempt to init Groq if Gemini keys are missing entirely but Groq key might be present
        # Since init_clients now only takes api_key (for Gemini), we need a way to init Groq if Gemini isn't used.
        # For simplicity, if GEMINI_API_KEYS is not set, we assume Gemini is not used.
        # We can call init_clients with a dummy key just to get Groq initialized if its key exists.
        # This assumes init_clients is robust enough to handle a dummy Gemini key if its purpose is only Groq init.
        # A cleaner way might be a separate init_groq_client() if this becomes too complex.
        try:
            _, st.session_state.groq_client = init_clients(api_key="GARBAGE_KEY_GEMINI_NOT_USED") # This call is primarily for Groq
            if st.session_state.groq_client:
                 st.sidebar.info("Groq client initialized (Gemini keys not provided in .env).")
            elif os.getenv("GROQ_API_KEY"): # Key was there but failed
                 st.sidebar.error("Groq client failed to initialize despite key in .env (Gemini keys not provided).")
            # No message if GROQ_API_KEY is also missing
        except Exception as e:
            st.sidebar.error(f"Error initializing Groq when Gemini keys are missing: {e}")

    # Sidebar for inputs
    with st.sidebar:
        st.header("Model Configuration") # Changed from API Configuration

        st.session_state.selected_model = st.radio(
            "Select Model:",
            ("Gemini", "Ollama"),
            index=0 if st.session_state.selected_model == "Gemini" else 1,  # Set index based on current session state
            help="Choose the model to use for generation. API keys for selected cloud models must be in .env."
        )

        # Ollama Specific Configuration
        if st.session_state.selected_model == "Ollama":
            st.subheader("Ollama Specific Configuration")
            st.session_state.ollama_max_prompt_chars = st.number_input(
                "Max content length for Ollama prompts (characters):",
                min_value=500,
                max_value=8000,
                value=st.session_state.get('ollama_max_prompt_chars', 2500),
                step=100,
                help="Maximum number of characters from paper content to include in prompts sent to Ollama. Adjust based on model context window and desired detail."
            )
            st.session_state.ollama_advanced_options_str = st.text_area(
                "Additional Ollama options (JSON format):",
                value=st.session_state.get('ollama_advanced_options_str', "{}"),
                height=100,
                help='Enter a JSON string for Ollama options, e.g., `{"temperature": 0.7, "num_predict": 100}`. These will be passed in the "options" field of the API request.'
            )

        # Gemini API key input and button removed.

        st.header("Research Configuration")

        # Base paper input
        input_method = st.radio("Base Paper Input Method:", ["URL", "Text"])

        if input_method == "URL":
            base_paper_url = st.text_input("Base Paper URL:")
            base_paper_content = st.text_area("Or paste paper excerpt:", height=200)
        else:
            base_paper_content = st.text_area("Paste Base Paper Content:", height=300)
            base_paper_url = ""

        # Research angle
        research_angle = st.text_area(
            "Your Research Angle/Critique:",
            placeholder="Describe how you want to build upon or diverge from the base paper...",
            height=150
        )

        # Optional seed keywords
        seed_keywords = st.text_input(
            "Seed Keywords (comma-separated):",
            placeholder="constitutional law, privacy rights, digital surveillance"
        )
        seed_keywords = [k.strip() for k in seed_keywords.split(",") if k.strip()]

        # Search parameters
        num_results = st.slider("Results per search:", 3, 10, 5)
        citation_style = st.selectbox(
            "Citation Style:",
            ["APA", "Bluebook", "OSCOLA", "ILI"],
            help="APA (7th ed.), Bluebook (21st ed.), OSCOLA (4th ed.), ILI format"
        )

        st.session_state.max_sources_to_analyze = st.number_input(
            "Max sources to analyze (0 for all found by crawler):",
            min_value=0,
            value=st.session_state.get('max_sources_to_analyze', 10),
            step=1,
            help="Maximum number of web search results to pass to downstream analysis agents (summaries, cases, relevance scoring). 0 means all (up to internal limits of those agents)."
        )

        st.subheader("Chat Configuration")  # Optional: Add a subheader for chat settings
        st.session_state.chat_retrieval_count = st.number_input(
            "Number of documents to retrieve for chat context:",
            min_value=1,
            max_value=10,
            value=st.session_state.get('chat_retrieval_count', 5),  # Default to 5 if not set
            step=1,
            help="How many relevant document excerpts should be retrieved from the vector store to provide context to the chat model."
        )

        # Run research button
        run_research = st.button("🔍 Start Research", type="primary", use_container_width=True)

    # Main content area
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None

    if run_research and (base_paper_content or base_paper_url) and research_angle:
        # Initialize Chroma Client if not already done
        if not st.session_state.chroma_client:
            try:
                st.session_state.chroma_client = chromadb.Client()  # In-memory client
                st.sidebar.info("Chroma client initialized successfully.")
            except Exception as e:
                st.sidebar.error(f"Chroma client initialization error: {e}")
                st.session_state.chroma_client = None  # Ensure it's None if failed
                # Optionally, stop if Chroma is critical
                # st.error("Failed to initialize Chroma client. Chat functionality will be unavailable.")
                # st.stop()

        # Get or create Chroma collection for this research session
        # This will effectively reset the collection for each new research run.
        # If persistence across runs (but still in-memory for the session) is needed,
        # this logic might need adjustment, e.g., naming the collection based on session ID
        # or checking if a collection with a specific name already exists.
        if st.session_state.chroma_client:
            try:
                # Using a timestamp-based or research-specific name could allow multiple research data sets
                # For now, using a generic name, effectively resetting for each "Start Research" click.
                collection_name = "legal_research_collection"
                st.session_state.chroma_collection = st.session_state.chroma_client.get_or_create_collection(
                    name=collection_name,
                    # embedding_function=st.session_state.ollama_embeddings # This causes error if client side embedding function not supported
                    # metadata={"hnsw:space": "cosine"} # Optional: specify distance metric if needed
                )
                st.sidebar.info(f"Chroma collection '{collection_name}' ready.")
            except Exception as e:
                st.sidebar.error(f"Chroma collection error: {e}")
                st.session_state.chroma_collection = None
                # Optionally, stop if Chroma collection is critical
                # st.error("Failed to get or create Chroma collection. Chat functionality will be unavailable.")
                # st.stop()

        # Check if Gemini client is initialized before running research, only if Gemini is selected
        if st.session_state.selected_model == "Gemini" and \
                (not st.session_state.get('clients_initialized', False) or not st.session_state.gemini_model):
            st.error(
                "Gemini model selected, but client is not initialized. Please enter your API key(s) in the sidebar, click 'Apply & Initialize Gemini Key(s)', and ensure it's successful before starting research.")
            st.stop()

        if st.session_state.selected_model == "Ollama":
            if not st.session_state.ollama_embeddings:
                st.session_state.ollama_embeddings = init_ollama_embeddings(
                    OLLAMA_EMBEDDING_BASE_URL, EMBEDDING_MODEL
                )
                if not st.session_state.ollama_embeddings:
                    st.error("Ollama Embeddings client could not be initialized. Chat functionality might be affected.")
                    # Optionally, you could stop execution if embeddings are critical:
                    # st.stop()

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create a container for thoughts and actions
        with st.expander("🧠 Agent Thoughts & Actions", expanded=True):
            thoughts_container = st.container()

        # Results containers
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("Research Results")
            results_container = st.container()

        with col2:
            st.header("Research Summary")
            summary_container = st.container()

        # Execute research pipeline
        with thoughts_container:
            st.markdown("### 🔍 Research Process")

        with st.spinner("Analyzing base paper..."):
            status_text.text("Agent 1: Extracting arguments...")
            progress_bar.progress(20)

            with thoughts_container:
                st.write("**Agent 1 - Argument Extraction:**")
                st.write("- Analyzing base paper content")
                st.write("- Identifying core thesis and legal concepts")
                st.write("- Understanding the new research angle")

            # If URL provided, note that actual fetching would require additional implementation
            if base_paper_url and not base_paper_content:
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
                    response = requests.get(base_paper_url, headers=headers, timeout=30)

                    if base_paper_url.lower().endswith('.pdf'):
                        # Handle PDF files
                        with io.BytesIO(response.content) as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            base_paper_content = ' '.join(page.extract_text() for page in pdf_reader.pages)
                    else:
                        # Handle HTML content
                        soup = BeautifulSoup(response.text, 'html.parser')

                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()

                        # Extract main content - prioritize article/main tags
                        main_content = soup.find(['article', 'main', 'div[role="main"]'])
                        if main_content:
                            base_paper_content = main_content.get_text(separator=' ', strip=True)
                        else:
                            # Fallback to body content
                            base_paper_content = soup.get_text(separator=' ', strip=True)

                    # Clean up extracted text
                    base_paper_content = ' '.join(base_paper_content.split())

                    if len(base_paper_content) < 100:
                        st.warning(
                            f"The extracted content from {base_paper_url} seems too short. Please verify the URL or paste the content directly.")

                except requests.RequestException as e:
                    st.error(f"Error fetching content from URL: {str(e)}")
                    base_paper_content = ""
                except Exception as e:
                    st.error(f"Error processing content from URL: {str(e)}")
                    base_paper_content = ""

            extracted_args = argument_extraction_agent(
                st.session_state.gemini_model,
                base_paper_content,
                research_angle
            )

            with summary_container:
                st.subheader("Extracted Arguments")
                st.write(f"**Core Thesis:** {extracted_args.get('core_thesis', 'N/A')}")
                st.write(f"**New Angle:** {extracted_args.get('new_angle', 'N/A')}")
                if extracted_args.get('key_concepts'):
                    st.write(f"**Key Concepts:** {', '.join(extracted_args.get('key_concepts', []))}")

            with thoughts_container:
                st.success(f"✓ Extracted {len(extracted_args.get('key_concepts', []))} key concepts")

        with st.spinner("Generating search keywords..."):
            status_text.text("Agent 2: Generating keywords...")
            progress_bar.progress(40)

            with thoughts_container:
                st.write("\n**Agent 2 - Keyword Generator:**")
                st.write("- Creating keyword permutations based on extracted concepts")
                st.write("- Simulating 'hit and try' search approach")
                st.write("- Generating broad and specific search terms")

            keywords = keyword_generator_agent(
                st.session_state.groq_client,
                extracted_args,
                seed_keywords
            )

            with summary_container:
                with st.expander("Generated Keywords"):
                    for kw in keywords[:10]:
                        st.write(f"• {kw}")

            with thoughts_container:
                st.success(f"✓ Generated {len(keywords)} search keywords")

        with st.spinner("Searching sources..."):
            status_text.text("Agent 3: Crawling sources...")
            progress_bar.progress(60)

            with thoughts_container:
                st.write("\n**Agent 3 - Source Crawler:**")
                st.write("- Searching SSRN and JSTOR for scholarly articles")
                st.write("- Looking for relevant case law and judgments")
                st.write("- Finding recent legal news and developments")
                st.write(f"- Using top {min(5, len(keywords))} keywords for search")

            search_results = source_crawler_agent(keywords, num_results)

            with thoughts_container:
                source_breakdown = {}
                for r in search_results:
                    source_type = r.get('source_type', 'unknown')
                    source_breakdown[source_type] = source_breakdown.get(source_type, 0) + 1

                st.success(f"✓ Found {len(search_results)} total sources:")
                for stype, count in source_breakdown.items():
                    st.write(f"  - {stype}: {count} sources")

        with st.spinner("Analyzing citations..."):
            status_text.text("Agent 4: Chaining citations...")
            progress_bar.progress(70)

            with thoughts_container:
                st.write("\n**Agent 4 - Citation Chainer:**")
                st.write("- Analyzing top papers for citation opportunities")
                st.write("- Identifying landmark cases and seminal works")
                st.write("- Building citation network")

            citation_suggestions = citation_chainer_agent(
                st.session_state.gemini_model,
                search_results[:5]
            )

            with thoughts_container:
                st.success(f"✓ Generated {len(citation_suggestions)} citation suggestions")

        with st.spinner("Extracting summaries and analyzing cases..."):
            status_text.text("Agent 5: Extracting summaries...")
            progress_bar.progress(80)

            with thoughts_container:
                st.write("\n**Agent 5 - Summary & Case Analysis:**")
                st.write("- Extracting main arguments from papers")
                st.write("- Analyzing case law for facts, issues, and judgments")
                st.write("- Identifying key findings")

            # Combine all results
            all_papers = search_results

            # Extract summaries
            all_papers = summary_extraction_agent(
                st.session_state.gemini_model,
                all_papers,
                extracted_args
            )

            # Analyze cases
            all_papers = case_analysis_agent(
                st.session_state.gemini_model,
                all_papers
            )

            with thoughts_container:
                st.success("✓ Completed content extraction")

            # Vector store population logic
            if st.session_state.ollama_embeddings and st.session_state.chroma_collection and all_papers:
                with st.spinner("Generating embeddings and updating vector store..."):
                    status_text.text("Embedding documents for chat...")
                    progress_bar.progress(85)

                    with thoughts_container:
                        st.write("\n**Vector Store Population:**")
                        st.write("- Preparing documents for embedding.")

                    documents_to_embed = []
                    metadatas_to_embed = []
                    ids_to_embed = []
                    text_contents_for_embedding = []

                    for i, paper in enumerate(all_papers):
                        content = ""
                        if paper.get('extracted_summary'):
                            content = paper['extracted_summary']
                        elif paper.get('raw_content'):
                            content = paper['raw_content'][:2000]
                        elif paper.get('snippet'):
                            content = paper['snippet']
                        else:
                            content = paper.get('title', '')

                        if not content.strip():
                            with thoughts_container:
                                st.warning(
                                    f"  - Skipping paper '{paper.get('title', 'Unknown Title')}' due to empty content for embedding.")
                            continue

                        text_contents_for_embedding.append(content)
                        metadata = {
                            "title": paper.get('title', 'N/A'),
                            "url": paper.get('url', 'N/A'),
                            "source_type": paper.get('source_type', 'N/A'),
                            "original_snippet": paper.get('snippet', '')[:500]
                        }
                        if paper.get('source_type') == 'case_law':
                            metadata["case_facts"] = paper.get('case_facts', 'N/A')
                            metadata["judgment"] = paper.get('judgment', 'N/A')
                        metadatas_to_embed.append(metadata)
                        ids_to_embed.append(
                            f"doc_{i}_{paper.get('url', paper.get('title', ''))[:50]}")

                    if text_contents_for_embedding:
                        try:
                            with thoughts_container:
                                st.write(
                                    f"- Generating embeddings for {len(text_contents_for_embedding)} document(s) using Ollama ({EMBEDDING_MODEL})...")
                            embeddings_vectors = st.session_state.ollama_embeddings.embed_documents(
                                text_contents_for_embedding)
                            with thoughts_container:
                                st.write(f"- Successfully generated {len(embeddings_vectors)} embedding vectors.")
                                st.write(
                                    f"- Adding {len(text_contents_for_embedding)} documents to Chroma collection '{st.session_state.chroma_collection.name}'.")
                            st.session_state.chroma_collection.add(
                                ids=ids_to_embed,
                                embeddings=embeddings_vectors,
                                metadatas=metadatas_to_embed,
                                documents=text_contents_for_embedding
                            )
                            with thoughts_container:
                                st.success(
                                    f"✓ Successfully added {st.session_state.chroma_collection.count()} documents to the vector store.")
                        except Exception as e:
                            with thoughts_container:
                                st.error(f"  - Error during embedding or adding to Chroma: {e}")
                            st.error(f"Failed to generate embeddings or update vector store: {e}")
                    else:
                        with thoughts_container:
                            st.warning("  - No documents found suitable for embedding.")
            elif not all_papers:
                with thoughts_container:
                    st.warning("\n**Vector Store Population:** No search results to embed.")
            elif not st.session_state.ollama_embeddings:
                with thoughts_container:
                    st.warning("\n**Vector Store Population:** Ollama embeddings client not available. Skipping embedding.")
            elif not st.session_state.chroma_collection:
                with thoughts_container:
                    st.warning("\n**Vector Store Population:** Chroma collection not available. Skipping embedding.")
            
        with st.spinner("Scoring relevance..."):
            status_text.text("Agent 6: Scoring papers...")
            progress_bar.progress(90)

            with thoughts_container:
                st.write("\n**Agent 6 - Relevance Scorer:**")
                st.write("- Scoring each source based on relevance to research angle")
                st.write("- Considering keyword matches and content alignment")
                st.write("- Ranking sources by score")

            # Score and rank
            scored_papers = relevance_scorer_agent(
                st.session_state.groq_client,
                all_papers[:20],  # Limit for performance
                extracted_args
            )

            with thoughts_container:
                score_distribution = {
                    "High (80-100)": len([p for p in scored_papers if p.get('relevance_score', 0) >= 80]),
                    "Medium (60-79)": len([p for p in scored_papers if 60 <= p.get('relevance_score', 0) < 80]),
                    "Low (40-59)": len([p for p in scored_papers if 40 <= p.get('relevance_score', 0) < 60]),
                }
                st.success("✓ Relevance scoring complete:")
                for range_name, count in score_distribution.items():
                    st.write(f"  - {range_name}: {count} sources")

        # Display results
        progress_bar.progress(100)
        status_text.text("Research complete!")

        with results_container:
            st.subheader(f"Found {len(scored_papers)} Relevant Sources")

            # Group results by type
            scholarly_papers = [p for p in scored_papers if p.get('source_type') == 'scholarly']
            case_law = [p for p in scored_papers if p.get('source_type') == 'case_law']
            news_articles = [p for p in scored_papers if p.get('source_type') == 'news']

            # Display by category
            if scholarly_papers:
                st.markdown("### 📚 Scholarly Articles")
                for idx, paper in enumerate(scholarly_papers[:5]):
                    with st.expander(f"📄 {paper['title'][:80]}... (Score: {paper.get('relevance_score', 'N/A')})"):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            # Display extracted summary if available
                            if paper.get('raw_content'):
                                st.write("**Summary:**")
                                summary = paper['raw_content'][:300] + "..." if len(paper['raw_content']) > 300 else \
                                    paper['raw_content']
                                st.write(summary)
                            else:
                                st.write(f"**Snippet:** {paper['snippet']}")

                            st.write(f"**Relevance:** {paper.get('relevance_reason', 'N/A')}")
                            if paper.get('key_insights'):
                                st.write(f"**Key Insights:** {paper['key_insights']}")
                            st.write(f"**Source:** {paper['source']} | **Keyword:** {paper.get('keyword_used', 'N/A')}")

                        with col2:
                            st.link_button("View Source", paper['url'], use_container_width=True)
                            if "pdf" in paper['url'].lower():
                                st.caption("📄 PDF Available")

                        # Citations in multiple formats
                        st.divider()
                        st.markdown("##### Citations")  # New sub-header for citations
                        st.code(f"APA: {format_citation(paper, 'APA')}", language="text")
                        st.code(f"Bluebook: {format_citation(paper, 'Bluebook')}", language="text")
                        st.code(f"OSCOLA: {format_citation(paper, 'OSCOLA')}", language="text")
                        st.code(f"ILI: {format_citation(paper, 'ILI')}", language="text")

            if case_law:
                st.markdown("### ⚖️ Legal Judgments & Case Law")
                for idx, case in enumerate(case_law[:5]):
                    with st.expander(f"⚖️ {case['title'][:80]}... (Score: {case.get('relevance_score', 'N/A')})"):
                        # Main case information
                        st.markdown("#### Case Analysis")

                        # Facts
                        st.markdown("**1. Facts:**")
                        st.write(case.get('case_facts', 'Facts not available'))

                        # Legal Issues
                        st.markdown("**2. Legal Issues:**")
                        issues = case.get('legal_issues', [])
                        if issues and isinstance(issues, list):
                            for issue in issues:
                                st.write(f"• {issue}")
                        else:
                            st.write("Legal issues not available")

                        # Arguments
                        st.markdown("**3. Arguments:**")
                        arguments = case.get('arguments', {})
                        if not isinstance(arguments, dict):
                            arguments = {"plaintiff": "Error: Invalid argument data",
                                         "defendant": "Error: Invalid argument data"}
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("*Plaintiff/Appellant:*")
                            st.write(arguments.get('plaintiff', 'Not available'))
                        with col2:
                            st.write("*Defendant/Respondent:*")
                            st.write(arguments.get('defendant', 'Not available'))

                        # Judgment
                        st.markdown("**4. Final Judgment:**")
                        st.write(case.get('judgment', 'Judgment not available'))

                        # Court Findings
                        st.markdown("**5. Court Findings:**")
                        st.write(case.get('court_findings', 'Court findings not available'))

                        # Additional info
                        st.divider()
                        st.markdown("**Additional Information:**")
                        st.write(f"*Relevance:* {case.get('relevance_reason', 'N/A')}")
                        st.write(f"*Source:* {case['source']} | *Keyword:* {case.get('keyword_used', 'N/A')}")

                        # View source button
                        st.link_button("View Full Case", case['url'], use_container_width=True)

                        # Citations in all formats
                        st.divider()
                        st.markdown("**Citations:**")

                        # Create tabs for different citation styles
                        tab1, tab2, tab3, tab4 = st.tabs(["APA", "Bluebook", "OSCOLA", "ILI"])

                        with tab1:
                            st.code(format_citation(case, "APA"), language="text")

                        with tab2:
                            st.code(format_citation(case, "Bluebook"), language="text")

                        with tab3:
                            st.code(format_citation(case, "OSCOLA"), language="text")

                        with tab4:
                            st.code(format_citation(case, "ILI"), language="text")

            if news_articles:
                st.markdown("### 📰 Recent Legal News & Developments")
                for idx, article in enumerate(news_articles[:5]):
                    with st.expander(f"📰 {article['title'][:80]}... (Score: {article.get('relevance_score', 'N/A')})"):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**Summary:** {article['snippet']}")
                            st.write(f"**Relevance:** {article.get('relevance_reason', 'N/A')}")
                            st.write(f"**Key Developments:** {article.get('key_insights', 'N/A')}")

                        with col2:
                            st.link_button("Read Article", article['url'], use_container_width=True)

                        # News citation
                        st.code(format_citation(article, citation_style), language="text")

            # Citation suggestions
            if citation_suggestions:
                st.markdown("### 🔗 Suggested Related Works")
                for suggestion in citation_suggestions[:5]:
                    st.write(f"• **{suggestion.get('title', 'N/A')}**")
                    st.write(f"  *Relevance:* {suggestion.get('relevance_reason', 'N/A')}")
                    st.write(f"  *From:* {suggestion.get('parent_paper', 'N/A')}")
                    if suggestion.get('search_terms'):
                        search_query = ' '.join(suggestion['search_terms'])
                        if st.button(f"Search for this", key=f"search_{suggestion.get('title', '')[:20]}"):
                            # Add to keywords for new search
                            st.session_state['additional_search'] = search_query

        # Export options
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            # Create research brief
            brief = f"""# Legal Research Brief
## Research Angle
{research_angle}

## Core Thesis of Base Paper
{extracted_args.get('core_thesis', 'N/A')}

## Research Findings

### Scholarly Articles
"""
            for paper in [p for p in scored_papers if p.get('source_type') == 'scholarly'][:5]:
                brief += f"\n#### {paper['title']}\n"
                brief += f"- Score: {paper.get('relevance_score', 'N/A')}\n"
                brief += f"- Summary: {paper.get('extracted_summary', paper.get('snippet', ''))[:200]}...\n"
                brief += f"- {paper.get('relevance_reason', 'N/A')}\n"
                brief += f"- Citation: {format_citation(paper, citation_style)}\n"

            brief += "\n### Case Law\n"
            for case in [p for p in scored_papers if p.get('source_type') == 'case_law'][:5]:
                brief += f"\n#### {case['title']}\n"
                brief += f"- Legal Principles: {case.get('key_insights', 'N/A')}\n"
                brief += f"- Citation: {format_citation(case, 'Bluebook')}\n"

            brief += "\n### Recent Developments\n"
            for article in [p for p in scored_papers if p.get('source_type') == 'news'][:5]:
                brief += f"\n#### {article['title']}\n"
                brief += f"- Key Points: {article.get('snippet', '')[:150]}...\n"

            st.download_button(
                label="📄 Download Research Brief",
                data=brief,
                file_name=f"research_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

        with col2:
            # Export as JSON
            export_data = {
                "research_angle": research_angle,
                "extracted_arguments": extracted_args,
                "keywords": keywords,
                "sources": scored_papers[:10]
            }
            st.download_button(
                label="📊 Export as JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with col3:
            # Save to session
            if st.button("💾 Save to Session"):
                if 'saved_research' not in st.session_state:
                    st.session_state.saved_research = []
                st.session_state.saved_research.append({
                    "timestamp": datetime.now(),
                    "angle": research_angle,
                    "results": scored_papers[:10]
                })
                st.success("Research saved to session!")

    elif run_research:
        st.error("Please provide both a base paper and research angle to begin.")

    st.divider()
    st.header("💬 Chat with Your Research")

    # Display chat history
    chat_display_container = st.container()
    with chat_display_container:
        for i, entry in enumerate(st.session_state.chat_history):
            if entry["role"] == "user":
                st.markdown(
                    f"<div style='text-align: right; margin-bottom: 8px;'><b>You:</b><br>{entry['content']}</div>",
                    unsafe_allow_html=True)
            else:  # assistant
                st.markdown(f"<div style='margin-bottom: 8px;'><b>Assistant:</b><br>{entry['content']}</div>",
                            unsafe_allow_html=True)

        # Add a small empty space at the bottom so the input is not too close to the last message
        st.empty()

    # Chat input
    user_chat_query = st.chat_input("Ask a question about your research results...")

    if user_chat_query:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_chat_query})

        # Ensure necessary components are available
        if not st.session_state.ollama_embeddings:
            error_msg = "Ollama embeddings client is not available. Cannot process chat query."
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            # Display error in chat and sidebar/main area for more visibility
            with chat_display_container:  # Assuming chat_display_container is in scope
                st.chat_message("assistant").error(error_msg)
            st.error(error_msg)  # Also show as a general error
            st.rerun()
        elif not st.session_state.chroma_collection:
            error_msg = "Chroma collection is not available. Cannot process chat query."
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            with chat_display_container:
                st.chat_message("assistant").error(error_msg)
            st.error(error_msg)
            st.rerun()
        elif not st.session_state.gemini_model:  # Assuming gemini_model is used for chat
            error_msg = "Gemini model is not initialized. Cannot process chat query."
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            with chat_display_container:
                st.chat_message("assistant").error(error_msg)
            st.error(error_msg)
            st.rerun()
        else:
            # Display user message in chat immediately
            # The loop for chat history will also pick this up on rerun, but this makes it feel more responsive
            # However, st.rerun() at the end handles the display update.
            # So, direct st.chat_message for user query is not strictly needed here if rerun is used.

            with st.spinner("Thinking..."):
                try:
                    # 1. Generate embedding for the user query
                    query_embedding = st.session_state.ollama_embeddings.embed_query(user_chat_query)

                    # 2. Query Chroma vector store
                    num_retrieved_docs = st.session_state.get('chat_retrieval_count', 5)
                    retrieved_results = st.session_state.chroma_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=num_retrieved_docs
                    )

                    # 3. Prepare context for Gemini
                    context_for_prompt = ""
                    if retrieved_results and retrieved_results.get('documents') and retrieved_results['documents'][0]:
                        retrieved_docs_texts = retrieved_results['documents'][0]
                        retrieved_metadatas = retrieved_results.get('metadatas', [[]])[0]

                        context_for_prompt += "Relevant excerpts from the research documents:\n\n"
                        for i, doc_text in enumerate(retrieved_docs_texts):
                            metadata_info = ""
                            if retrieved_metadatas and i < len(retrieved_metadatas):
                                meta = retrieved_metadatas[i]
                                title = meta.get('title', 'Unknown Title')
                                url = meta.get('url', 'N/A')
                                source_type = meta.get('source_type', 'Unknown Source')
                                metadata_info = f" (Source Document: '{title}', Type: {source_type}, URL: {url})"

                            context_for_prompt += f"Excerpt {i + 1}{metadata_info}:\n"
                            context_for_prompt += f"```text\n{doc_text}\n```\n\n"

                    else:
                        context_for_prompt = "No specific documents matching your query were found in the current research collection. You can ask general questions or questions about the research process itself."

                    # 4. Construct prompt for Gemini
                    prompt = f"""You are a helpful AI Legal Research Assistant. Your role is to answer the user's questions based ONLY on the provided 'Context from research'.
Do not use any external knowledge or information outside of this context.
If the context is insufficient or does not contain the answer, clearly state that the provided research documents do not cover the query.
Be concise and directly address the user's question.

User Question: {user_chat_query}

Context from research:
{context_for_prompt}

Assistant's Answer:
"""
                    # 5. Send prompt to Gemini
                    if st.session_state.gemini_api_keys_list:  # Ensure API keys are available
                        active_key = st.session_state.gemini_api_keys_list[st.session_state.current_gemini_key_index]
                        genai.configure(api_key=active_key, transport='rest')
                    else:  # Should not happen if Gemini model is initialized, but as a safeguard
                        raise ValueError("Gemini API keys are not configured.")

                    # Using the existing st.session_state.gemini_model
                    gemini_response = st.session_state.gemini_model.generate_content(prompt)
                    assistant_response = gemini_response.text

                except Exception as e:
                    assistant_response = f"Sorry, I encountered an error while processing your request: {e}"
                    st.error(f"Chat processing error: {e}")  # Log the error for debugging

            # 6. Add Gemini response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

            # 7. Rerun to update the chat display immediately
            # This will re-render the chat_display_container with the new history
            st.rerun()

    # This ensures the chat_display_container is defined before being potentially used in error conditions above
    # This is a re-iteration of the chat display logic from the previous step,
    # but st.rerun() makes it so it doesn't strictly need to be duplicated if user_chat_query is true.
    # However, to be safe and handle the case where user_chat_query is false (initial load/no new input),
    # the display logic should be outside the `if user_chat_query:` block.
    # The previous step correctly placed the display logic outside.
    # The `with chat_display_container:` for error messages inside `if user_chat_query:` might need adjustment
    # if `chat_display_container` is not always defined when an error occurs before its definition.
    # For now, assuming `chat_display_container` is defined earlier in the UI rendering flow.

    # Correct placement of chat display (should be outside the `if user_chat_query` block as in previous step)
    # This part is already handled by the previous plan step.
    # The `st.rerun()` will cause the existing chat display logic to refresh.

    # Display saved research
    if 'saved_research' in st.session_state and st.session_state.saved_research:
        st.divider()
        st.header("Saved Research Sessions")
        for idx, research in enumerate(st.session_state.saved_research):
            with st.expander(f"Session {idx + 1}: {research['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(f"**Angle:** {research['angle']}")
                st.write(f"**Results:** {len(research['results'])} sources found")


if __name__ == "__main__":
    main()