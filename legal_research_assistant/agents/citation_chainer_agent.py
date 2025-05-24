import spacy
import re
from legal_research_assistant.agents.source_crawler_agent import SourceCrawlerAgent # Assuming direct import path

class CitationChainerAgent:
    """
    A class to perform rudimentary citation chaining from crawled sources.
    It extracts potential author-year patterns from abstracts and re-searches them.
    """
    def __init__(self, source_crawler: SourceCrawlerAgent, model_name="en_core_web_sm"):
        """
        Initializes the agent with a SourceCrawlerAgent instance and loads a spaCy model.

        Args:
            source_crawler: An instance of SourceCrawlerAgent.
            model_name (str): The name of the spaCy model to load.
        """
        self.source_crawler = source_crawler
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # TODO: Replace print with logging
            print(f"spaCy model '{model_name}' not found. Please download it by running:")
            print(f"python -m spacy download {model_name}")
            print(f"Falling back to a blank English model ('en'). Citation chaining quality will be significantly affected.")
            self.nlp = spacy.blank("en")
            if not self.nlp.has_pipe("sentencizer"):
                 self.nlp.add_pipe("sentencizer")

        # Basic regex to find patterns like (Author, Year) or Author (Year)
        # This is a simplified regex and might need refinement.
        # It looks for:
        # - One or more capitalized words (Author), optionally followed by " et al."
        # - A comma (optional)
        # - Whitespace
        # - An opening parenthesis (optional)
        # - A year (19xx or 20xx)
        # - A closing parenthesis (optional)
        self.citation_regex = re.compile(r"([A-Z][A-Za-z'-]+(?: et al\.)?,?\s*(?:\(?(19\d{2}|20\d{2})\)?))")

    def _extract_search_terms_from_abstract(self, abstract: str) -> list[str]:
        """
        Extracts potential search terms (like "Author Year") from an abstract.
        """
        if not abstract or not isinstance(abstract, str) or not abstract.strip():
            return []

        doc = self.nlp(abstract)
        search_terms = []

        for sent in doc.sents:
            matches = self.citation_regex.finditer(sent.text)
            for match in matches:
                # The first group of the regex captures the "Author, Year" like string
                # We can further process this, e.g., remove parentheses, commas for a cleaner search term
                potential_citation_text = match.group(1)
                
                # Basic cleaning: remove parentheses, extract main author and year
                # E.g., "Smith (2020)" -> "Smith 2020"
                # E.g., "Jones et al., 1999" -> "Jones et al 1999"
                term = potential_citation_text.replace('(', '').replace(')', '')
                term = term.replace(',', '') # Remove comma before year if present
                # Further refinement could be to split author and year, but for Tavily, combined might work.
                search_terms.append(term.strip())
        
        return list(set(search_terms)) # Return unique search terms

    def chain_citations(self, crawled_sources: list[dict]) -> list[dict]:
        """
        Performs citation chaining from a list of crawled sources.

        Args:
            crawled_sources: A list of dictionaries, where each dictionary is a source
                             from SourceCrawlerAgent and includes an 'abstract'.

        Returns:
            A list of new_potential_sources (dictionaries in the same format as
            SourceCrawlerAgent output), which are potential "chained" citations.
        """
        if not self.source_crawler:
            # TODO: Replace print with logging
            print("CitationChainerAgent: SourceCrawlerAgent not provided. Cannot perform chaining.")
            return []
        if not self.nlp:
            # TODO: Replace print with logging
            print("CitationChainerAgent: spaCy model not loaded. Cannot perform chaining.")
            return []

        all_new_potential_sources = []
        all_extracted_search_terms = []

        for source in crawled_sources:
            abstract = source.get('abstract')
            if not abstract:
                continue
            
            extracted_terms = self._extract_search_terms_from_abstract(abstract)
            if extracted_terms:
                all_extracted_search_terms.extend(extracted_terms)

        # Deduplicate all extracted search terms before searching
        unique_search_terms = list(set(all_extracted_search_terms))
        
        if not unique_search_terms:
            print("CitationChainerAgent: No potential citation search terms extracted from abstracts.")
            return []

        print(f"CitationChainerAgent: Extracted {len(unique_search_terms)} unique potential citation search terms for re-searching.")
        # Example: print a few terms
        # print(f"   Example terms: {unique_search_terms[:5]}")


        # Search for these terms using the source crawler
        # We use max_results_per_keyword=1 to get the most relevant hit for each "citation"
        if unique_search_terms:
            print("   - Searching SSRN for extracted citation terms...")
            ssrn_chained_sources = self.source_crawler.search_ssrn(unique_search_terms, max_results_per_keyword=1)
            all_new_potential_sources.extend(ssrn_chained_sources)
            
            print("   - Searching JSTOR for extracted citation terms...")
            jstor_chained_sources = self.source_crawler.search_jstor(unique_search_terms, max_results_per_keyword=1)
            all_new_potential_sources.extend(jstor_chained_sources)

        # Deduplicate based on URL
        final_deduplicated_sources = []
        seen_urls = set()
        for source in all_new_potential_sources:
            url = source.get('url')
            if url and url not in seen_urls:
                final_deduplicated_sources.append(source)
                seen_urls.add(url)
        
        print(f"CitationChainerAgent: Found {len(final_deduplicated_sources)} new unique potential sources from citation chaining.")
        return final_deduplicated_sources
