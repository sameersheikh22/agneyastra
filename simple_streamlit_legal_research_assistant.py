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
    gemini_model = genai.GenerativeModel('gemini-pro')

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
        # Fallback extraction based on the research angle
        return {
            "core_thesis": f"Analysis of {research_angle[:100]}...",
            "key_concepts": ["AI regulation", "risk management", "government competencies", "tort liability"],
            "new_angle": research_angle,
            "research_directions": ["Comparative analysis of AI regulations", "Tort liability frameworks for AI"]
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
        # Fallback keywords based on extracted arguments
        base_keywords = []

        # Add angle-based keywords
        if extracted_args.get('new_angle'):
            angle_words = extracted_args['new_angle'].split()[:5]
            base_keywords.append(' '.join(angle_words))

        # Add concept-based keywords
        concepts = extracted_args.get('key_concepts', [])
        if concepts:
            base_keywords.extend([
                f"{concepts[0]} legal framework",
                f"{concepts[0]} case law",
                f"{concepts[0]} {concepts[1] if len(concepts) > 1 else 'regulation'}"
            ])

        # Add seed keywords
        if seed_keywords:
            base_keywords.extend(seed_keywords[:3])

        # Generate variations
        fallback_keywords = [
            "AI regulation tort liability",
            "artificial intelligence governance risks",
            "AI legal framework government competencies",
            "machine learning regulation challenges",
            "algorithmic accountability legal",
            "AI liability frameworks comparative",
            "artificial intelligence tort law",
            "AI risk management legal policy"
        ]

        # Combine and return
        all_keywords = base_keywords + fallback_keywords
        return all_keywords[:15]


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
        {{"score": 85, "reason": "Directly addresses AI regulation through tort liability", "insights": "Proposes differential liability framework for AI systems"}}
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
            paper['key_insights'] = scoring.get('insights', 'Relevant to AI regulation research')
            scored_papers.append(paper)
        except Exception as e:
            # Fallback scoring based on keyword matching
            title_lower = paper.get('title', '').lower()
            snippet_lower = paper.get('snippet', '').lower()
            angle_lower = research_context.get('new_angle', '').lower()

            # Simple keyword-based scoring
            score = 50  # Base score
            reasons = []

            # Check for AI/artificial intelligence mentions
            if 'artificial intelligence' in title_lower or 'ai' in title_lower:
                score += 20
                reasons.append("mentions AI")

            # Check for regulation/governance mentions
            if any(word in title_lower for word in ['regulat', 'govern', 'policy', 'law']):
                score += 15
                reasons.append("discusses regulation")

            # Check for risk mentions
            if 'risk' in title_lower or 'challenge' in title_lower:
                score += 10
                reasons.append("addresses risks")

            # Check for tort/liability mentions
            if 'tort' in angle_lower and ('liabil' in title_lower or 'tort' in title_lower):
                score += 15
                reasons.append("covers liability aspects")

            paper['relevance_score'] = min(score, 95)  # Cap at 95
            paper['relevance_reason'] = f"Keyword analysis: {', '.join(reasons) if reasons else 'general relevance'}"
            paper[
                'key_insights'] = f"May provide insights on {' and '.join(reasons[:2]) if reasons else 'AI regulation'}"
            scored_papers.append(paper)

    # Sort by relevance score
    scored_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    return scored_papers


# Add new Summary Extraction Agent after the citation chainer agent
def summary_extraction_agent(gemini_model, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract meaningful summaries from paper content"""
    for paper in papers[:10]:  # Limit for performance
        if paper.get('raw_content') and len(paper['raw_content']) > 100:
            prompt = f"""
            Extract a concise summary from this academic paper content:

            Title: {paper['title']}
            Content: {paper['raw_content'][:1500]}

            Provide:
            1. Main argument/thesis (1-2 sentences)
            2. Key findings or principles (2-3 bullet points)
            3. Relevance to AI regulation and liability

            Format as JSON with keys: main_argument, key_findings, ai_relevance
            """

            try:
                response = gemini_model.generate_content(prompt)
                summary_data = json.loads(response.text)
                paper['extracted_summary'] = summary_data.get('main_argument', '')
                paper['key_findings'] = summary_data.get('key_findings', [])
                paper['ai_relevance'] = summary_data.get('ai_relevance', '')
            except:
                # Fallback to snippet
                paper['extracted_summary'] = paper.get('snippet', '')[:200]
                paper['key_findings'] = []
                paper['ai_relevance'] = "Relevant to AI governance research"
        else:
            paper['extracted_summary'] = paper.get('snippet', '')[:200]

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
        citation_style = st.selectbox("Citation Style:", ["APA", "Bluebook", "Custom"])

        # Run research button
        run_research = st.button("🔍 Start Research", type="primary", use_container_width=True)

    # Main content area
    if run_research and (base_paper_content or base_paper_url) and research_angle:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Results containers
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("Research Results")
            results_container = st.container()

        with col2:
            st.header("Research Summary")
            summary_container = st.container()

        # Execute research pipeline
        with st.spinner("Analyzing base paper..."):
            status_text.text("Agent 1: Extracting arguments...")
            progress_bar.progress(20)

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

        with st.spinner("Generating search keywords..."):
            status_text.text("Agent 2: Generating keywords...")
            progress_bar.progress(40)

            keywords = keyword_generator_agent(
                st.session_state.groq_client,
                extracted_args,
                seed_keywords
            )

            with summary_container:
                with st.expander("Generated Keywords"):
                    for kw in keywords[:10]:
                        st.write(f"• {kw}")

        with st.spinner("Searching sources..."):
            status_text.text("Agent 3: Crawling sources...")
            progress_bar.progress(60)

            search_results = source_crawler_agent(keywords, num_results)

        with st.spinner("Analyzing citations..."):
            status_text.text("Agent 4: Chaining citations...")
            progress_bar.progress(70)

            citation_suggestions = citation_chainer_agent(
                st.session_state.gemini_model,
                search_results[:5]
            )

        with st.spinner("Scoring relevance..."):
            status_text.text("Agent 5: Scoring papers...")
            progress_bar.progress(90)

            # Combine all results
            all_papers = search_results

            # Score and rank
            scored_papers = relevance_scorer_agent(
                st.session_state.groq_client,
                all_papers[:20],  # Limit for performance
                extracted_args
            )

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

                        # Citation
                        st.code(format_citation(paper, citation_style), language="text")

            if case_law:
                st.markdown("### ⚖️ Legal Judgments & Case Law")
                for idx, case in enumerate(case_law[:5]):
                    with st.expander(f"⚖️ {case['title'][:80]}... (Score: {case.get('relevance_score', 'N/A')})"):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**Case Summary:** {case['snippet']}")
                            st.write(f"**Relevance:** {case.get('relevance_reason', 'N/A')}")
                            st.write(f"**Legal Principles:** {case.get('key_insights', 'N/A')}")

                        with col2:
                            st.link_button("View Case", case['url'], use_container_width=True)

                        # Legal citation
                        st.code(format_citation(case, "Bluebook"), language="text")

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

## Top Sources
"""
            for paper in scored_papers[:10]:
                brief += f"\n### {paper['title']}\n"
                brief += f"- Score: {paper.get('relevance_score', 'N/A')}\n"
                brief += f"- {paper.get('relevance_reason', 'N/A')}\n"
                brief += f"- Citation: {format_citation(paper, citation_style)}\n"

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