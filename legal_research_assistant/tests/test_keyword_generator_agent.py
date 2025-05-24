import pytest
from unittest.mock import patch, MagicMock, call
from legal_research_assistant.agents.keyword_generator_agent import KeywordGeneratorAgent
from collections import Counter # Used by the agent, good to be aware of
import spacy # For spec in MagicMock

# --- Fixtures (adapted from test_argument_extraction_agent) ---

@pytest.fixture
def mock_token_kg_factory():
    """Factory to create a mock spaCy Token for KeywordGeneratorAgent."""
    def _factory(text="word", lemma="lemma", pos="NOUN", is_stop=False, is_punct=False):
        token = MagicMock(spec=spacy.tokens.Token)
        token.text = text
        token.text.lower = text.lower # For direct access
        token.lemma_ = lemma.lower()
        token.pos_ = pos
        token.is_stop = is_stop
        token.is_punct = is_punct
        # KeywordGeneratorAgent also uses token.lemma_.lower() implicitly via list appends
        return token
    return _factory

@pytest.fixture
def mock_doc_kg_factory(mock_token_kg_factory):
    """Factory to create a mock spaCy Doc for KeywordGeneratorAgent."""
    def _factory(tokens_data=None): # tokens_data is a list of dicts for token properties
        doc_instance = MagicMock(spec=spacy.tokens.Doc)
        if tokens_data is None:
            tokens_data = [{'text': 'default', 'lemma': 'default', 'pos': 'NOUN'}]
        
        mock_tokens = [mock_token_kg_factory(**td) for td in tokens_data]
        doc_instance.__iter__.return_value = iter(mock_tokens)
        return doc_instance
    return _factory

@pytest.fixture
def mock_spacy_nlp_kg(mock_doc_kg_factory):
    """Mocks the spaCy nlp object for KeywordGeneratorAgent."""
    nlp = MagicMock(spec=spacy.language.Language)
    
    # This side_effect will be overridden in tests for specific token sequences
    nlp.side_effect = lambda text: mock_doc_kg_factory() 
    
    nlp.Defaults = MagicMock()
    nlp.Defaults.stop_words = {'is', 'a', 'the', 'and', 'for', 'in', 'of', 'to', 'use', 'uses'} 
    nlp.has_pipe = MagicMock(return_value=True)
    nlp.add_pipe = MagicMock()
    return nlp

@pytest.fixture
def mock_spacy_blank_nlp_kg(mock_doc_kg_factory):
    """Mocks a blank spaCy model for KeywordGeneratorAgent fallback testing."""
    nlp_blank = MagicMock(spec=spacy.language.Language)
    nlp_blank.side_effect = lambda text: mock_doc_kg_factory(tokens_data=[]) # Blank processes to empty/simple doc
    nlp_blank.Defaults = MagicMock()
    nlp_blank.Defaults.stop_words = set()
    nlp_blank.has_pipe = MagicMock(return_value=False)
    nlp_blank.add_pipe = MagicMock()
    return nlp_blank

# --- Test Cases for Initialization ---

def test_kg_agent_initialization_success(mock_spacy_nlp_kg):
    with patch('spacy.load', return_value=mock_spacy_nlp_kg) as mock_load:
        agent = KeywordGeneratorAgent(model_name='en_core_web_sm_test_kg')
        mock_load.assert_called_once_with('en_core_web_sm_test_kg')
        assert agent.nlp is mock_spacy_nlp_kg

def test_kg_agent_initialization_primary_fails_fallback_sm_succeeds(mock_spacy_nlp_kg):
    def side_effect_spacy_load(model_name, **kwargs):
        if model_name == 'non_existent_model_kg':
            raise OSError("Model 'non_existent_model_kg' not found")
        elif model_name == "en_core_web_sm":
            return mock_spacy_nlp_kg
        raise ValueError(f"Unexpected model load attempt: {model_name}")

    with patch('spacy.load', side_effect=side_effect_spacy_load) as mock_load, \
         patch('builtins.print'): # Mock print to silence warnings during test
        agent = KeywordGeneratorAgent(model_name='non_existent_model_kg')
        expected_load_calls = [call('non_existent_model_kg'), call('en_core_web_sm')]
        mock_load.assert_has_calls(expected_load_calls, any_order=False)
        assert agent.nlp is mock_spacy_nlp_kg

def test_kg_agent_initialization_all_fallbacks_to_blank(mock_spacy_blank_nlp_kg):
    def side_effect_spacy_load(model_name, **kwargs):
        if model_name == 'non_existent_model_kg' or model_name == "en_core_web_sm":
            raise OSError(f"Model '{model_name}' not found")
        raise ValueError(f"Unexpected model load attempt: {model_name}")

    with patch('spacy.load', side_effect=side_effect_spacy_load) as mock_load, \
         patch('spacy.blank', return_value=mock_spacy_blank_nlp_kg) as mock_blank, \
         patch('builtins.print'):
        agent = KeywordGeneratorAgent(model_name='non_existent_model_kg')
        mock_blank.assert_called_once_with("en")
        assert agent.nlp is mock_spacy_blank_nlp_kg
        # Check if sentencizer (and tagger, though agent doesn't explicitly add tagger to blank) was added
        mock_spacy_blank_nlp_kg.add_pipe.assert_any_call("sentencizer")


# --- Test Cases for generate_keywords Method ---

def test_generate_keywords_valid_summary(mock_spacy_nlp_kg, mock_doc_kg_factory):
    # Define token data for the specific summary text
    summary_text = "Legal AI uses advanced algorithms for research in intellectual property."
    tokens_data = [
        {'text': 'Legal', 'lemma': 'legal', 'pos': 'ADJ', 'is_stop': False},
        {'text': 'AI', 'lemma': 'ai', 'pos': 'PROPN', 'is_stop': False},
        {'text': 'uses', 'lemma': 'use', 'pos': 'VERB', 'is_stop': True}, # Stopword
        {'text': 'advanced', 'lemma': 'advanced', 'pos': 'ADJ', 'is_stop': False},
        {'text': 'algorithms', 'lemma': 'algorithm', 'pos': 'NOUN', 'is_stop': False},
        {'text': 'for', 'lemma': 'for', 'pos': 'ADP', 'is_stop': True},    # Stopword
        {'text': 'research', 'lemma': 'research', 'pos': 'NOUN', 'is_stop': False},
        {'text': 'in', 'lemma': 'in', 'pos': 'ADP', 'is_stop': True},      # Stopword
        {'text': 'intellectual', 'lemma': 'intellectual', 'pos': 'ADJ', 'is_stop': False},
        {'text': 'property', 'lemma': 'property', 'pos': 'NOUN', 'is_stop': False},
        {'text': '.', 'lemma': '.', 'pos': 'PUNCT', 'is_punct': True},
    ]
    mock_spacy_nlp_kg.side_effect = lambda text: mock_doc_kg_factory(tokens_data=tokens_data if text == summary_text else [])

    with patch('spacy.load', return_value=mock_spacy_nlp_kg):
        agent = KeywordGeneratorAgent(top_n_terms=5, max_keywords=15) # Control parameters
        extracted_args = {'research_angle_summary': summary_text}
        
        keywords = agent.generate_keywords(extracted_args)
        
        # Expected key terms (lemmatized, from specific POS, non-stop, non-punct):
        # legal, ai, advanced, algorithm, research, intellectual, property
        # top_n_terms=5 will pick 5 of these based on mocked (equal) frequency.
        # Let's assume for this mock, they are picked in order of appearance if frequencies are flat.
        # So, 'legal', 'ai', 'advanced', 'algorithm', 'research'
        
        assert isinstance(keywords, list)
        
        # Check for single terms (top 5 based on our assumption for this mock)
        # Note: The agent's Counter might result in a different order for "most_common" if all have count 1.
        # For stable testing, we should ensure our mock provides distinct frequencies or control Counter's output.
        # For simplicity, let's check presence for now.
        expected_single_terms = ['legal', 'ai', 'advanced', 'algorithm', 'research', 'intellectual', 'property']
        present_single_terms = [term for term in keywords if ' ' not in term]

        # Check that the top_n_terms were selected and are present
        # This is tricky because Counter's order for same-frequency items is not guaranteed.
        # We will check if the returned single keywords are a subset of expected.
        for term in present_single_terms:
            assert term in expected_single_terms
        
        # Check for combinations (these are sorted alphabetically by the agent)
        # If 'ai', 'algorithm', 'advanced', 'legal', 'research' were the top 5 for combinations:
        assert "ai algorithm" in keywords # Example 2-word combo (sorted)
        assert "advanced legal research" in keywords # Example 3-word combo (sorted)
        
        assert len(keywords) <= 15 # max_keywords


def test_generate_keywords_empty_summary(mock_spacy_nlp_kg):
    with patch('spacy.load', return_value=mock_spacy_nlp_kg):
        agent = KeywordGeneratorAgent()
        extracted_args = {'research_angle_summary': ''}
        keywords = agent.generate_keywords(extracted_args)
        assert keywords == ["Missing or invalid research angle summary"]

def test_generate_keywords_no_key_terms(mock_spacy_nlp_kg, mock_doc_kg_factory):
    # All tokens are stopwords or wrong POS or punctuation
    tokens_data = [
        {'text': 'is', 'lemma': 'be', 'pos': 'AUX', 'is_stop': True},
        {'text': 'a', 'lemma': 'a', 'pos': 'DET', 'is_stop': True},
        {'text': '.', 'lemma': '.', 'pos': 'PUNCT', 'is_punct': True},
    ]
    summary_text = "is a ."
    mock_spacy_nlp_kg.side_effect = lambda text: mock_doc_kg_factory(tokens_data=tokens_data if text == summary_text else [])

    with patch('spacy.load', return_value=mock_spacy_nlp_kg):
        agent = KeywordGeneratorAgent()
        extracted_args = {'research_angle_summary': summary_text}
        keywords = agent.generate_keywords(extracted_args)
        assert keywords == ["No key terms extracted from summary"]

def test_generate_keywords_max_keywords_limit(mock_spacy_nlp_kg, mock_doc_kg_factory):
    # Many unique, processable terms
    tokens_data = [ {'text': f'word{i}', 'lemma': f'word{i}', 'pos': 'NOUN'} for i in range(15) ]
    summary_text = " ".join([td['text'] for td in tokens_data])
    mock_spacy_nlp_kg.side_effect = lambda text: mock_doc_kg_factory(tokens_data=tokens_data if text == summary_text else [])

    with patch('spacy.load', return_value=mock_spacy_nlp_kg):
        # top_n_terms=7, max_keywords=10.
        # 7 single terms. 7C2 = 21 two-word. 7C3 = 35 three-word.
        # Output should be capped at 10.
        agent = KeywordGeneratorAgent(top_n_terms=7, max_keywords=10) 
        extracted_args = {'research_angle_summary': summary_text}
        keywords = agent.generate_keywords(extracted_args)
        assert len(keywords) == 10

def test_generate_keywords_deduplication(mock_spacy_nlp_kg, mock_doc_kg_factory):
    tokens_data = [ # "apple", "banana", "apple" -> "apple", "banana"
        {'text': 'apple', 'lemma': 'apple', 'pos': 'NOUN'},
        {'text': 'banana', 'lemma': 'banana', 'pos': 'NOUN'},
        {'text': 'apple', 'lemma': 'apple', 'pos': 'NOUN'}, # Duplicate lemma
    ]
    summary_text = "apple banana apple"
    mock_spacy_nlp_kg.side_effect = lambda text: mock_doc_kg_factory(tokens_data=tokens_data if text == summary_text else [])
    
    with patch('spacy.load', return_value=mock_spacy_nlp_kg):
        agent = KeywordGeneratorAgent(top_n_terms=2) # apple, banana
        extracted_args = {'research_angle_summary': summary_text}
        keywords = agent.generate_keywords(extracted_args)
        
        # Expected: 'apple', 'banana', 'apple banana' (sorted combination)
        assert 'apple' in keywords
        assert 'banana' in keywords
        assert 'apple banana' in keywords
        assert len(keywords) == 3 # Check total count to infer deduplication of singles
        
        # Check that 'apple' appears only once as a single keyword
        single_keywords = [k for k in keywords if ' ' not in k]
        assert single_keywords.count('apple') == 1


def test_generate_keywords_fewer_terms_than_top_n(mock_spacy_nlp_kg, mock_doc_kg_factory):
    tokens_data = [
        {'text': 'one', 'lemma': 'one', 'pos': 'NOUN'},
        {'text': 'two', 'lemma': 'two', 'pos': 'ADJ'},
    ]
    summary_text = "one two"
    mock_spacy_nlp_kg.side_effect = lambda text: mock_doc_kg_factory(tokens_data=tokens_data if text == summary_text else [])

    with patch('spacy.load', return_value=mock_spacy_nlp_kg):
        agent = KeywordGeneratorAgent(top_n_terms=5) # Request 5, but only 2 available
        extracted_args = {'research_angle_summary': summary_text}
        keywords = agent.generate_keywords(extracted_args)
        
        # Expected: 'one', 'two', 'one two' (sorted combination)
        assert 'one' in keywords
        assert 'two' in keywords
        assert 'one two' in keywords # 2C2 = 1 combination
        # No 3-word combinations possible
        assert len(keywords) == 3

def test_generate_keywords_no_combinations_if_few_terms(mock_spacy_nlp_kg, mock_doc_kg_factory):
    tokens_data = [{'text': 'soloterm', 'lemma': 'soloterm', 'pos': 'NOUN'}]
    summary_text = "soloterm"
    mock_spacy_nlp_kg.side_effect = lambda text: mock_doc_kg_factory(tokens_data=tokens_data if text == summary_text else [])

    with patch('spacy.load', return_value=mock_spacy_nlp_kg):
        agent = KeywordGeneratorAgent(top_n_terms=5)
        extracted_args = {'research_angle_summary': summary_text}
        keywords = agent.generate_keywords(extracted_args)
        assert keywords == ['soloterm'] # Only the single term, no combinations


def test_generate_keywords_input_not_dict(mock_spacy_nlp_kg):
    with patch('spacy.load', return_value=mock_spacy_nlp_kg):
        agent = KeywordGeneratorAgent()
        keywords = agent.generate_keywords("this is a string, not a dict")
        assert keywords == ["Missing or invalid research angle summary"] # Or handle error differently

def test_generate_keywords_missing_summary_key(mock_spacy_nlp_kg):
    with patch('spacy.load', return_value=mock_spacy_nlp_kg):
        agent = KeywordGeneratorAgent()
        extracted_args = {'other_key': 'Some text'}
        keywords = agent.generate_keywords(extracted_args)
        assert keywords == ["Missing or invalid research angle summary"]

```
