import pytest
from unittest.mock import patch, MagicMock, call
from legal_research_assistant.agents.citation_chainer_agent import CitationChainerAgent
from legal_research_assistant.agents.source_crawler_agent import SourceCrawlerAgent # For spec
import spacy # For spec

# --- Fixtures ---

@pytest.fixture
def mock_sentence_cca_factory():
    """Factory for creating mock spaCy sentence objects for CCA."""
    def _factory(text="A sentence about Smith (2020)."):
        sent = MagicMock(spec=spacy.tokens.Span)
        sent.text = text
        return sent
    return _factory

@pytest.fixture
def mock_doc_cca_factory(mock_sentence_cca_factory):
    """Factory for creating mock spaCy Doc objects for CCA."""
    def _factory(text="Abstract text. Mentions Perez (2019) and Lee et al. (2022)."):
        doc = MagicMock(spec=spacy.tokens.Doc)
        # Simple sentence splitting for testing regex on sentences
        sentences_texts = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences_texts and text.strip(): # Handle case where text has no periods but is not empty
            sentences_texts = [text.strip()]
            
        doc.sents = [mock_sentence_cca_factory(text=st) for st in sentences_texts]
        return doc
    return _factory

@pytest.fixture
def mock_spacy_nlp_cca(mock_doc_cca_factory):
    """Mocks the spaCy nlp object for CitationChainerAgent."""
    nlp = MagicMock(spec=spacy.language.Language)
    nlp.side_effect = lambda text: mock_doc_cca_factory(text=text)
    nlp.has_pipe = MagicMock(return_value=True)
    nlp.add_pipe = MagicMock()
    return nlp

@pytest.fixture
def mock_spacy_blank_nlp_cca(mock_doc_cca_factory):
    """Mocks a blank spaCy model for CCA fallback testing."""
    nlp_blank = MagicMock(spec=spacy.language.Language)
    nlp_blank.side_effect = lambda text: mock_doc_cca_factory(text=text) # Blank model still processes
    nlp_blank.has_pipe = MagicMock(return_value=False) # Simulate no pipes initially
    nlp_blank.add_pipe = MagicMock()
    return nlp_blank

@pytest.fixture
def mock_source_crawler():
    """Mocks SourceCrawlerAgent for CCA."""
    crawler = MagicMock(spec=SourceCrawlerAgent)
    # Default to returning empty lists, can be overridden per test
    crawler.search_ssrn = MagicMock(return_value=[])
    crawler.search_jstor = MagicMock(return_value=[])
    return crawler

# --- Test Cases for Initialization ---

def test_cca_initialization_success(mock_spacy_nlp_cca, mock_source_crawler):
    with patch('spacy.load', return_value=mock_spacy_nlp_cca) as mock_load:
        agent = CitationChainerAgent(source_crawler=mock_source_crawler, model_name='en_test_model_cca')
        mock_load.assert_called_once_with('en_test_model_cca')
        assert agent.nlp is mock_spacy_nlp_cca
        assert agent.source_crawler is mock_source_crawler

def test_cca_initialization_spacy_fallback_to_sm(mock_spacy_nlp_cca, mock_source_crawler):
    def side_effect_spacy_load(model_name, **kwargs):
        if model_name == 'primary_fails_cca':
            raise OSError("Primary model load failed for CCA")
        elif model_name == 'en_core_web_sm':
            return mock_spacy_nlp_cca # Successfully load fallback
        raise ValueError(f"Unexpected model load: {model_name}")

    with patch('spacy.load', side_effect=side_effect_spacy_load) as mock_load, \
         patch('builtins.print'): # Silence print warnings
        agent = CitationChainerAgent(source_crawler=mock_source_crawler, model_name='primary_fails_cca')
        expected_calls = [call('primary_fails_cca'), call('en_core_web_sm')]
        mock_load.assert_has_calls(expected_calls)
        assert agent.nlp is mock_spacy_nlp_cca

def test_cca_initialization_spacy_fallback_to_blank(mock_spacy_blank_nlp_cca, mock_source_crawler):
    def side_effect_spacy_load(model_name, **kwargs):
        if model_name == 'primary_fails_cca' or model_name == 'en_core_web_sm':
            raise OSError(f"Model load failed for {model_name}")
        raise ValueError(f"Unexpected model load: {model_name}")

    with patch('spacy.load', side_effect=side_effect_spacy_load) as mock_load, \
         patch('spacy.blank', return_value=mock_spacy_blank_nlp_cca) as mock_blank, \
         patch('builtins.print'):
        agent = CitationChainerAgent(source_crawler=mock_source_crawler, model_name='primary_fails_cca')
        mock_blank.assert_called_once_with("en")
        assert agent.nlp is mock_spacy_blank_nlp_cca
        mock_spacy_blank_nlp_cca.add_pipe.assert_called_with("sentencizer")


# --- Test Cases for _extract_search_terms_from_abstract ---
# Note: Testing private methods directly is generally okay for utility-like private methods.

def test_extract_search_terms_multiple_patterns(mock_spacy_nlp_cca, mock_source_crawler):
    with patch('spacy.load', return_value=mock_spacy_nlp_cca): # nlp needed for the method
        agent = CitationChainerAgent(source_crawler=mock_source_crawler) 
    
    abstract_text = "Work by Smith (2020). Also, (Jones et al., 1999). See Miller (2010) for details. And (Gamma,2005)."
    # Agent's regex: r"([A-Z][A-Za-z'-]+(?: et al\.)?,?\s*(?:\(?(19\d{2}|20\d{2})\)?))"
    # Expected after cleaning by the method: "Smith 2020", "Jones et al 1999", "Miller 2010", "Gamma 2005"
    
    extracted_terms = agent._extract_search_terms_from_abstract(abstract_text)
    
    assert len(extracted_terms) == 4
    assert "Smith 2020" in extracted_terms
    assert "Jones et al 1999" in extracted_terms
    assert "Miller 2010" in extracted_terms
    assert "Gamma 2005" in extracted_terms

def test_extract_search_terms_no_patterns(mock_spacy_nlp_cca, mock_source_crawler):
    with patch('spacy.load', return_value=mock_spacy_nlp_cca):
        agent = CitationChainerAgent(source_crawler=mock_source_crawler)
    abstract_text = "This abstract contains no standard citation formats."
    extracted_terms = agent._extract_search_terms_from_abstract(abstract_text)
    assert extracted_terms == []

def test_extract_search_terms_empty_abstract(mock_spacy_nlp_cca, mock_source_crawler):
    with patch('spacy.load', return_value=mock_spacy_nlp_cca):
        agent = CitationChainerAgent(source_crawler=mock_source_crawler)
    extracted_terms = agent._extract_search_terms_from_abstract("")
    assert extracted_terms == []


# --- Test Cases for chain_citations ---

MOCK_SSRN_RESULT_A = {"title": "SSRN Chained A", "url": "http://ssrn.com/abs/A", "abstract": "Abstract A", "source": "SSRN (via Tavily)", "score": 0.9}
MOCK_JSTOR_RESULT_B = {"title": "JSTOR Chained B", "url": "http://jstor.org/stable/B", "abstract": "Abstract B", "source": "JSTOR (via Tavily)", "score": 0.88}
MOCK_SSRN_RESULT_C_DUPLICATE_URL_A = {"title": "SSRN Chained C", "url": "http://ssrn.com/abs/A", "abstract": "Abstract C", "source": "SSRN (via Tavily)", "score": 0.85}


def test_chain_citations_finds_and_returns_new_sources(mock_spacy_nlp_cca, mock_source_crawler):
    with patch('spacy.load', return_value=mock_spacy_nlp_cca):
        agent = CitationChainerAgent(source_crawler=mock_source_crawler)

    crawled_sources = [
        {"url": "http://example.com/source1", "abstract": "Mentions AuthorA (2021) and AuthorB et al. (2022)."}
    ]
    
    # Mock SourceCrawler responses based on extracted terms
    def mock_ssrn_search(keywords, max_results_per_keyword):
        if any("AuthorA 2021" in kw for kw in keywords):
            return [MOCK_SSRN_RESULT_A]
        return []
        
    def mock_jstor_search(keywords, max_results_per_keyword):
        if any("AuthorB et al 2022" in kw for kw in keywords):
            return [MOCK_JSTOR_RESULT_B]
        return []

    mock_source_crawler.search_ssrn.side_effect = mock_ssrn_search
    mock_source_crawler.search_jstor.side_effect = mock_jstor_search
    
    chained_results = agent.chain_citations(crawled_sources)
    
    assert len(chained_results) == 2
    assert MOCK_SSRN_RESULT_A in chained_results
    assert MOCK_JSTOR_RESULT_B in chained_results
    
    # Check calls to source_crawler (example for one)
    # Extracted terms would be ["AuthorA 2021", "AuthorB et al 2022"]
    mock_source_crawler.search_ssrn.assert_any_call(["AuthorA 2021", "AuthorB et al 2022"], max_results_per_keyword=1)
    mock_source_crawler.search_jstor.assert_any_call(["AuthorA 2021", "AuthorB et al 2022"], max_results_per_keyword=1)


def test_chain_citations_deduplicates_results(mock_spacy_nlp_cca, mock_source_crawler):
    with patch('spacy.load', return_value=mock_spacy_nlp_cca):
        agent = CitationChainerAgent(source_crawler=mock_source_crawler)

    # Abstract leads to "Term1 (2020)" and "Term2 (2021)"
    crawled_sources = [{"url": "id0", "abstract": "Source mentions Term1 (2020) and also Term2 (2021)."}]

    # Both terms, when searched, return the same paper from different sources or calls
    mock_source_crawler.search_ssrn.side_effect = lambda keywords, max_results_per_keyword: \
        [MOCK_SSRN_RESULT_A] if any(t in kw for t in ["Term1 2020", "Term2 2021"]) else []
    mock_source_crawler.search_jstor.side_effect = lambda keywords, max_results_per_keyword: \
        [MOCK_SSRN_RESULT_A] if any(t in kw for t in ["Term1 2020", "Term2 2021"]) else [] # Simulating it finds same URL

    results = agent.chain_citations(crawled_sources)
    assert len(results) == 1 # Should be deduplicated by URL
    assert results[0]['url'] == MOCK_SSRN_RESULT_A['url']


def test_chain_citations_no_patterns_in_abstracts(mock_spacy_nlp_cca, mock_source_crawler):
    with patch('spacy.load', return_value=mock_spacy_nlp_cca):
        agent = CitationChainerAgent(source_crawler=mock_source_crawler)
    crawled_sources = [{"url": "id1", "abstract": "This abstract has no citation patterns."}]
    
    results = agent.chain_citations(crawled_sources)
    assert results == []
    mock_source_crawler.search_ssrn.assert_not_called()
    mock_source_crawler.search_jstor.assert_not_called()


def test_chain_citations_patterns_found_no_new_search_results(mock_spacy_nlp_cca, mock_source_crawler):
    with patch('spacy.load', return_value=mock_spacy_nlp_cca):
        agent = CitationChainerAgent(source_crawler=mock_source_crawler)
    # mock_source_crawler is already set to return [] for searches by default
    crawled_sources = [{"url": "id1", "abstract": "References FakeAuthor (2000)."}]
    
    results = agent.chain_citations(crawled_sources)
    assert results == []
    # Assert that search methods were called (because "FakeAuthor 2000" was extracted)
    mock_source_crawler.search_ssrn.assert_called_with(["FakeAuthor 2000"], max_results_per_keyword=1)
    mock_source_crawler.search_jstor.assert_called_with(["FakeAuthor 2000"], max_results_per_keyword=1)


def test_chain_citations_empty_input_crawled_sources(mock_spacy_nlp_cca, mock_source_crawler):
    with patch('spacy.load', return_value=mock_spacy_nlp_cca):
        agent = CitationChainerAgent(source_crawler=mock_source_crawler)
    results = agent.chain_citations([])
    assert results == []


def test_chain_citations_source_with_missing_abstract(mock_spacy_nlp_cca, mock_source_crawler):
    with patch('spacy.load', return_value=mock_spacy_nlp_cca):
        agent = CitationChainerAgent(source_crawler=mock_source_crawler)
    crawled_sources = [{"url": "id1", "title": "Paper without abstract"}] # No 'abstract' key
    results = agent.chain_citations(crawled_sources)
    assert results == []
    mock_source_crawler.search_ssrn.assert_not_called()

def test_chain_citations_source_with_empty_abstract_string(mock_spacy_nlp_cca, mock_source_crawler):
    with patch('spacy.load', return_value=mock_spacy_nlp_cca):
        agent = CitationChainerAgent(source_crawler=mock_source_crawler)
    crawled_sources = [{"url": "id1", "abstract": ""}] # Empty abstract
    results = agent.chain_citations(crawled_sources)
    assert results == []
    mock_source_crawler.search_ssrn.assert_not_called()


def test_chain_citations_nlp_model_failed_to_load(mock_source_crawler):
    # Simulate NLP model failing completely in __init__
    with patch('spacy.load', side_effect=OSError("Cannot load any model for CCA")) as mock_load, \
         patch('spacy.blank', side_effect=OSError("Cannot even load blank model")) as mock_blank, \
         patch('builtins.print') as mock_print: # Capture print messages
        agent = CitationChainerAgent(source_crawler=mock_source_crawler)
        # agent.nlp will be None or a non-functional mock if spacy.blank also fails
        # For this test, let's assume it ends up as something that evaluates to False or is None
        # The agent's code checks 'if not self.nlp:'

    # To ensure self.nlp is None, we might need to control the __init__ more directly
    # For now, let's assume the above patch setup makes self.nlp effectively unusable
    # A more direct way:
    agent.nlp = None # Manually set nlp to None to simulate complete load failure

    crawled_sources = [{"url": "id1", "abstract": "Some Author (2023)"}]
    with patch('builtins.print') as mock_print_method_call:
        results = agent.chain_citations(crawled_sources)
        assert results == []
        mock_print_method_call.assert_any_call("CitationChainerAgent: spaCy model not loaded. Cannot perform chaining.")

```
