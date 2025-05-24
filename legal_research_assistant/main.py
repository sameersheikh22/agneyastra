from agents import (
    ArgumentExtractionAgent,
    KeywordGeneratorAgent,
    SourceCrawlerAgent,
    CitationChainerAgent,
    RelevanceScorerAgent
)
from utils import format_output
from utils.persistence_utils import save_results # Import the save_results function
from config import get_api_key # Import the function to get API keys
import re # For sanitizing filename
import datetime # For timestamp in filename

def main():
    """
    Main function to orchestrate the legal research assistant.
    """
    # a. Define more descriptive placeholder inputs
    base_paper_text = """
    The concept of "fair use" in copyright law is a judicially created doctrine that has been codified in Section 107 of the Copyright Act.
    It permits the limited use of copyrighted material without acquiring permission from the rights holders.
    Courts typically consider four factors: (1) the purpose and character of the use, including whether such use is of a commercial nature or is for nonprofit educational purposes;
    (2) the nature of the copyrighted work; (3) the amount and substantiality of the portion used in relation to the copyrighted work as a whole; and
    (4) the effect of the use upon the potential market for or value of the copyrighted work.
    Recent rulings have emphasized the transformative nature of the use as a key aspect of the first factor.
    This doctrine is essential for balancing the interests of copyright holders with the public's interest in the dissemination of information.
    """
    research_angle_text = """
    Investigate the application of the 'transformative use' principle within the fair use doctrine,
    particularly in cases involving digital media and software.
    Focus on how courts interpret 'transformativeness' and its impact on the overall fair use analysis.
    """
    initial_paper_for_citation_chain = "https://example.com/initial_paper_for_citation_chain" # Remains a placeholder for now

    print("Starting Legal Research Assistant...")
    print(f"Base Paper (snippet): {base_paper_text[:100]}...")
    print(f"Research Angle: {research_angle_text}")
    print("-" * 30)

    # b. Instantiate each agent
    arg_extractor = ArgumentExtractionAgent()
    keyword_generator = KeywordGeneratorAgent()
    # SourceCrawlerAgent and CitationChainerAgent will be instantiated later,
    # once the Tavily API key is fetched and SourceCrawlerAgent is initialized.
    # source_crawler = SourceCrawlerAgent() # Old instantiation
    # citation_chainer = CitationChainerAgent() # Old instantiation
    relevance_scorer = RelevanceScorerAgent()

    # c. Call agent methods in sequence
    print("\n1. Extracting Arguments...")
    extracted_args_data = arg_extractor.extract_arguments(base_paper_text, research_angle_text)
    print(f"   - Base Paper Summary: {extracted_args_data.get('base_paper_summary', 'N/A')}")
    print(f"   - Research Angle Summary: {extracted_args_data.get('research_angle_summary', 'N/A')}")

    print("\n2. Generating Keywords...")
    # KeywordGeneratorAgent now expects the full extracted_args_data dictionary
    # which contains 'research_angle_summary'.
    keywords = keyword_generator.generate_keywords(extracted_args_data)
    print(f"   - Generated keywords: {keywords}")

    print("\n3. Crawling Sources (SSRN via Tavily)...")
    tavily_api_key = get_api_key("TAVILY_API_KEY")
    # Re-instantiate SourceCrawlerAgent with the API key
    # Note: In a more complex app, API keys might be managed globally or via a config object
    # passed to agents, rather than re-instantiating or passing directly in main.
    # For this step, direct instantiation is fine.
    if not tavily_api_key or "YOUR_TAVILY_KEY_HERE" in tavily_api_key:
        print("   - Tavily API Key not found or is a placeholder. Skipping SSRN search.")
        crawled_sources = []
    else:
        # Instantiate SourceCrawlerAgent with the API key
        source_crawler = SourceCrawlerAgent(api_key=tavily_api_key)
        crawled_sources = source_crawler.search_ssrn(keywords) # Use the new method
        print(f"   - Found {len(crawled_sources)} articles from SSRN (via Tavily).")
        if crawled_sources:
            print("   --- Example SSRN Articles ---")
            for i, article in enumerate(crawled_sources[:2]): # Print first 2 articles as examples
                print(f"     Article {i+1}:")
                print(f"       Title: {article.get('title', 'N/A')}")
                print(f"       URL: {article.get('url', 'N/A')}")
                print(f"       Abstract Snippet: {article.get('abstract', 'N/A')[:150]}...") # Print snippet
                print(f"       Source: {article.get('source', 'N/A')}")
            print("   --- End of Examples ---")
    # Store SSRN results
    ssrn_articles = crawled_sources 
    
    # --- Add JSTOR Search ---
    jstor_articles = []
    if not tavily_api_key or "YOUR_TAVILY_KEY_HERE" in tavily_api_key:
        print("   - Tavily API Key not found or is a placeholder. Skipping JSTOR search.")
    else:
        # source_crawler should already be instantiated if API key was valid
        if hasattr(source_crawler, 'search_jstor'): # Check if source_crawler is the updated version
            print("\n3b. Crawling Sources (JSTOR via Tavily)...")
            jstor_articles = source_crawler.search_jstor(keywords)
            print(f"   - Found {len(jstor_articles)} articles from JSTOR (via Tavily).")
            if jstor_articles:
                print("   --- Example JSTOR Articles ---")
                for i, article in enumerate(jstor_articles[:2]): # Print first 2 articles
                    print(f"     Article {i+1}:")
                    print(f"       Title: {article.get('title', 'N/A')}")
                    print(f"       URL: {article.get('url', 'N/A')}")
                    print(f"       Abstract Snippet: {article.get('abstract', 'N/A')[:150]}...")
                    print(f"       Source: {article.get('source', 'N/A')}")
                print("   --- End of Examples ---")
        else:
            # This case handles if source_crawler wasn't updated, perhaps due to API key issue earlier
            print("   - SourceCrawlerAgent does not have search_jstor method (API key might have been missing).")

    # Combine crawled sources (SSRN + JSTOR)
    all_crawled_sources = []
    if ssrn_articles:
        all_crawled_sources.extend(ssrn_articles)
    if jstor_articles:
        all_crawled_sources.extend(jstor_articles)
    
    # Deduplicate crawled sources by URL
    unique_crawled_by_url = {article['url']: article for article in all_crawled_sources}
    all_crawled_sources = list(unique_crawled_by_url.values())
    print(f"\n   - Total unique articles from keyword search (SSRN & JSTOR): {len(all_crawled_sources)}")


    print("\n4. Performing Citation Chaining...")
    # For now, citation chaining uses a placeholder initial paper.
    # If it were to use crawled sources, we'd need to adapt its input.
    chained_sources_output = citation_chainer.chain_citations(initial_paper_for_citation_chain)
    print(f"   - Found {len(chained_sources_output)} items from citation chaining (placeholder output).")
    # For consistency, ensure chained_sources_output is a list of dicts if it's to be combined
    # Assuming chained_sources_output is a list of strings for now as per its placeholder.
    # We will treat them as separate from the structured articles for scoring.
    
    # Combine all found items for relevance scoring:
    # For scoring, we need a consistent structure. Let's assume citation chainer
    # eventually returns structured data similar to crawled sources.
    # For now, we'll primarily score the crawled sources.
    # If chained_sources_output were dicts: all_found_sources.extend(chained_sources_output)
    all_found_sources_for_scoring = list(all_crawled_sources) # Primarily score crawled articles

    # If chained_sources are just strings (URLs/titles), they might need different handling or scoring
    # Initialize source_crawler and citation_chainer after API key check
    source_crawler_instance = None # To hold the instance
    if not tavily_api_key or "YOUR_TAVILY_KEY_HERE" in tavily_api_key:
        # Already printed a warning about API key for SSRN/JSTOR search
        pass # source_crawler_instance remains None
    else:
        source_crawler_instance = source_crawler # source_crawler was instantiated inside the 'else' block for Tavily key

    # --- Citation Chaining Step ---
    chained_articles = []
    if source_crawler_instance: # Only proceed if SourceCrawlerAgent was initialized (API key was present)
        citation_chainer = CitationChainerAgent(source_crawler=source_crawler_instance)
        print("\n4b. Performing Citation Chaining (using crawled SSRN & JSTOR abstracts)...")
        if all_crawled_sources:
            chained_articles = citation_chainer.chain_citations(all_crawled_sources)
            if chained_articles:
                print(f"   --- Found {len(chained_articles)} potential new articles from citation chaining ---")
                # Example print for a couple of chained articles
                for i, article in enumerate(chained_articles[:2]):
                    print(f"     Chained Article {i+1}: {article.get('title', 'N/A')} ({article.get('url', 'N/A')})")
            else:
                print("   --- No new articles found from citation chaining. ---")
        else:
            print("   --- No crawled sources available to perform citation chaining. ---")
    else:
        print("\n4b. Skipping Citation Chaining as SourceCrawlerAgent was not initialized (likely missing API key).")

    # Combine all sources: crawled (SSRN, JSTOR) + chained
    final_sources_to_score = list(all_crawled_sources) # Start with unique crawled sources
    if chained_articles:
        final_sources_to_score.extend(chained_articles)
        # Deduplicate again after adding chained articles
        final_sources_by_url = {source['url']: source for source in final_sources_to_score if source.get('url')}
        final_sources_to_score = list(final_sources_by_url.values())
        print(f"\n   - Total unique sources after combining crawled and chained: {len(final_sources_to_score)}")


    print(f"\n   - Total items to be considered for relevance scoring: {len(final_sources_to_score)}")

    print("\n5. Scoring Relevance of Sources...")
    research_summary_for_scoring = extracted_args_data.get("research_angle_summary", research_angle_text)
    scored_sources = relevance_scorer.score_sources(final_sources_to_score, research_summary_for_scoring)
    print(f"   - Scored {len(scored_sources)} sources.")

    # d. Call format_output with the final list of scored sources
    print("\n6. Formatting Output (and printing top results with new scores)...")
    # The format_output function will also need to be aware of these new scores if it's
    # to display them. For now, we'll print a few examples directly in main.py
    # before calling format_output.
    
    print("\n--- Top Scored Sources (Examples) ---")
    for i, source in enumerate(scored_sources[:3]): # Print top 3 examples
        print(f"  Source {i+1}:")
        print(f"    Title: {source.get('title', 'N/A')}")
        print(f"    URL: {source.get('url', 'N/A')}")
        print(f"    Source Type: {source.get('source', 'N/A')}")
        print(f"    Tavily Score: {source.get('score', 0.0):.4f}") # Original Tavily score
        print(f"    Similarity to Angle: {source.get('similarity_to_research_angle', 0.0):.4f}")
        print(f"    Final Relevance Score: {source.get('final_relevance_score', 0.0):.4f}")
        # print(f"    Abstract Snippet: {source.get('abstract', 'N/A')[:150]}...")
    print("--- End of Top Scored Examples ---")

    # The existing format_output might not show these new scores unless updated.
    # For this subtask, we've printed them above.
    formatted_output_str = format_output(scored_sources) 

    # e. Print the formatted output from the utility function
    print("\n" + "=" * 30)
    print("Full Formatted Research Report (from format_output utility):")
    print("=" * 30)
    print(formatted_output_str)

    # Example of fetching and printing an API key (for demonstration)
    print("-" * 30)
    print("Demonstrating API Key Access (for dev purposes):")
    tavily_api_key = get_api_key("TAVILY_API_KEY")
    if tavily_api_key and "YOUR_" not in tavily_api_key: # Basic check if it's a real key
        print(f"   - Tavily API Key: {tavily_api_key[:10]}... (partially hidden)")
    else:
        print(f"   - Tavily API Key: {tavily_api_key} (placeholder or not set)")
    
    google_api_key = get_api_key("GOOGLE_AI_API_KEY")
    if google_api_key and "YOUR_" not in google_api_key:
        print(f"   - Google AI API Key: {google_api_key[:10]}... (partially hidden)")
    else:
        print(f"   - Google AI API Key: {google_api_key} (placeholder or not set)")
    print("-" * 30)

    # --- Save final results ---
    print("\n7. Saving Results...")
    results_to_save = {
        "base_paper_summary": extracted_args_data.get("base_paper_summary"),
        "research_angle_summary": extracted_args_data.get("research_angle_summary"),
        "keywords_used": keywords,
        "scored_sources": scored_sources # This now contains combined & scored sources
    }
    
    # Create a filename_prefix (using a simplified version of the example)
    research_angle_short = "_".join(research_angle_summary.split()[:5]) if research_angle_summary else "research"
    sanitized_prefix = re.sub(r'[^a-zA-Z0-9_]', '', research_angle_short)
    if not sanitized_prefix: 
        sanitized_prefix = "research_session"
    
    # The save_results function in persistence_utils now handles the timestamp internally
    # So we just pass the sanitized_prefix.
    saved_filepath = save_results(filename_prefix=sanitized_prefix, data_to_save=results_to_save)
    if saved_filepath:
        print(f"   - Successfully saved results to: {saved_filepath}")
    else:
        print("   - Failed to save results.")


if __name__ == "__main__":
    main()
