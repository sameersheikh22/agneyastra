import streamlit as st
import requests
from typing import List, Dict, Any
import json
import re
from urllib.parse import urlparse, quote
import time
from datetime import datetime
import google.generativeai as genai
from groq import Groq

# Page config
st.set_page_config(
    page_title="Legal Research Companion",
    page_icon="⚖️",
    layout="wide"
)


# Initialize API clients
def init_clients():
    """Initialize API clients with provided keys"""
    # Gemini
    genai.configure(api_key="AIzaSyBvaCZAq2bJkLgdA1kuY_IBLE6TkzP7k1k")
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')

    # Groq
    groq_client = Groq(api_key="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")

    return gemini_model, groq_client


# Agent 1: Argument Extraction Agent
def argument_extraction_agent(gemini_model, base_paper_content: str, research_angle: str) -> Dict[str, str]:
    """Extract core thesis and identify new research direction"""
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

    try:
        response = gemini_model.generate_content(prompt)
        # Clean response text
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        return result
    except Exception as e:
        # Dynamic fallback based on actual content
        angle_words = research_angle.split()[:10]
        content_words = base_paper_content.split()[:50]

        # Extract key terms from the input
        key_terms = []
        for word in angle_words + content_words:
            if len(word) > 4 and word.lower() not in ['this', 'that', 'these', 'those', 'which', 'where', 'when']:
                key_terms.append(word)

        return {
            "core_thesis": f"Analysis of legal aspects related to {' '.join(angle_words[:5])}",
            "key_concepts": list(set(key_terms[:5])) if key_terms else ["legal analysis", "research", "regulation"],
            "new_angle": research_angle,
            "research_directions": [
                f"Comparative analysis of {angle_words[0] if angle_words else 'topic'}",
                f"Legal framework for {' '.join(angle_words[:3]) if angle_words else 'subject matter'}"
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
    tavily_key = "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"
    tavily_url = "https://api.tavily.com/search"

    # Serper API search
    serper_key = "3611eaea5638a59ec95b6329077ddd9c8a71ece3"
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


# Agent 4: Citation Chainer Agent
def citation_chainer_agent(gemini_model, top_papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract and follow citation trails"""
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

        Format as JSON list with keys: title, relevance_reason, search_terms
        """

        try:
            response = gemini_model.generate_content(prompt)
            citations = json.loads(response.text)
            for citation in citations:
                citation['parent_paper'] = paper['title']
                chained_citations.append(citation)
        except:
            pass

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

    for paper in papers:
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


# Add new Summary Extraction Agent after the citation chainer agent
def summary_extraction_agent(gemini_model, papers: List[Dict[str, Any]], research_context: Dict[str, str]) -> List[
    Dict[str, Any]]:
    """Extract meaningful summaries from paper content"""
    research_topic = research_context.get('new_angle', 'legal research')

    for paper in papers[:10]:  # Limit for performance
        if paper.get('raw_content') and len(paper['raw_content']) > 100:
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

            try:
                response = gemini_model.generate_content(prompt)
                summary_data = json.loads(response.text)
                paper['extracted_summary'] = summary_data.get('main_argument', '')
                paper['key_findings'] = summary_data.get('key_findings', [])
                paper['topic_relevance'] = summary_data.get('topic_relevance', '')
            except:
                # Fallback to snippet
                paper['extracted_summary'] = paper.get('snippet', '')[:200]
                paper['key_findings'] = []
                paper['topic_relevance'] = f"Relevant to {research_topic}"
        else:
            paper['extracted_summary'] = paper.get('snippet', '')[:200]

    return papers


# Agent for Case Analysis
def case_analysis_agent(gemini_model, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyzes legal case documents from a list of papers to extract structured information.

    For each paper identified as 'case_law', this function uses a Gemini model
    to extract facts, legal issues, arguments from both sides, the judgment,
    and court findings. It updates the paper dictionary with this information.
    """
    for paper in papers:
        if paper.get('source_type') == 'case_law':
            try:
                # Determine content to analyze, prioritizing raw_content
                content_to_analyze = ""
                raw_content = paper.get('raw_content', '')
                snippet_content = paper.get('snippet', '')

                if raw_content and len(raw_content) > 200:
                    content_to_analyze = raw_content
                elif snippet_content:
                    content_to_analyze = snippet_content
                else:
                    content_to_analyze = paper.get('title', '') # Fallback to title if no content

                # Truncate content to avoid overly long prompts
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

                response = gemini_model.generate_content(prompt)
                
                # Clean response text if needed (though Gemini usually provides clean JSON with proper prompting)
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                analysis_results = json.loads(response_text)

                paper['case_facts'] = analysis_results.get('case_facts', "Not available")
                paper['legal_issues'] = analysis_results.get('legal_issues', [])
                paper['arguments'] = analysis_results.get('arguments', {"plaintiff": "Not available", "defendant": "Not available"})
                paper['judgment'] = analysis_results.get('judgment', "Not available")
                paper['court_findings'] = analysis_results.get('court_findings', "Not available")

            except json.JSONDecodeError as e:
                st.warning(f"Error decoding JSON for paper '{paper.get('title', 'Unknown Title')}': {e}. Raw response: {response_text[:200]}")
                # Set defaults if parsing fails to ensure keys exist if expected later
                paper['case_facts'] = paper.get('case_facts', "Not available after parsing error")
                paper['legal_issues'] = paper.get('legal_issues', [])
                paper['arguments'] = paper.get('arguments', {"plaintiff": "Not available", "defendant": "Not available"})
                paper['judgment'] = paper.get('judgment', "Not available after parsing error")
                paper['court_findings'] = paper.get('court_findings', "Not available after parsing error")
            except Exception as e:
                st.warning(f"Error analyzing case '{paper.get('title', 'Unknown Title')}': {e}")
                # Set defaults if API call or other processing fails
                paper['case_facts'] = paper.get('case_facts', "Not available after API/processing error")
                paper['legal_issues'] = paper.get('legal_issues', [])
                paper['arguments'] = paper.get('arguments', {"plaintiff": "Not available", "defendant": "Not available"})
                paper['judgment'] = paper.get('judgment', "Not available after API/processing error")
                paper['court_findings'] = paper.get('court_findings', "Not available after API/processing error")
                
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

    # Initialize clients
    if 'clients_initialized' not in st.session_state:
        with st.spinner("Initializing AI agents..."):
            st.session_state.gemini_model, st.session_state.groq_client = init_clients()
            st.session_state.clients_initialized = True

    # Sidebar for inputs
    with st.sidebar:
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

        # Run research button
        run_research = st.button("🔍 Start Research", type="primary", use_container_width=True)

    # Main content area
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None

    if run_research and (base_paper_content or base_paper_url) and research_angle:
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
                base_paper_content = f"[Content from {base_paper_url}]"

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
                        with st.expander("View Citations"):
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