import pytest
from unittest.mock import patch, MagicMock, call
from legal_research_assistant.agents.argument_extraction_agent import ArgumentExtractionAgent
import spacy # Import spacy to check for spacy.blank and spacy.tokens.Doc later

# --- Fixtures ---

@pytest.fixture
def mock_token_factory():
    """Factory to create a mock spaCy Token."""
    def _factory(text="word", lemma="lemma", is_stop=False, is_alpha=True):
        token = MagicMock(spec=spacy.tokens.Token)
        token.text = text
        token.text.lower = text.lower # For direct access
        token.lemma_ = lemma.lower()
        token.is_stop = is_stop
        token.is_alpha = is_alpha
        return token
    return _factory

@pytest.fixture
def mock_sentence_factory(mock_token_factory):
    """Factory to create a mock spaCy Span (sentence)."""
    def _factory(text="This is a sentence.", tokens=None):
        sent = MagicMock(spec=spacy.tokens.Span)
        sent.text = text
        if tokens is None:
            # Create mock tokens from the sentence text if not provided
            words_in_sent = text.split()
            tokens = [mock_token_factory(text=w, lemma=w) for w in words_in_sent]
        sent.__iter__.return_value = iter(tokens)
        return sent
    return _factory

@pytest.fixture
def mock_doc_factory(mock_sentence_factory, mock_token_factory):
    """Factory to create a mock spaCy Doc."""
    def _factory(text="Doc text. Another sentence.", sentences_texts=None, doc_tokens=None):
        doc_instance = MagicMock(spec=spacy.tokens.Doc)
        
        if sentences_texts is None:
            sentences_texts = [s.strip() for s in text.split('.') if s.strip()]
        
        mock_sents = []
        all_tokens_for_doc_iter = []

        for sent_text in sentences_texts:
            # Create tokens for this sentence for sentence iteration
            words_in_sent = sent_text.split()
            tokens_for_sent_iter = [mock_token_factory(text=w, lemma=w, is_stop=(w.lower() in {"is", "a", "the"})) for w in words_in_sent]
            sent_mock = mock_sentence_factory(text=sent_text, tokens=tokens_for_sent_iter)
            mock_sents.append(sent_mock)
            # all_tokens_for_doc_iter.extend(tokens_for_sent_iter) # For doc.__iter__

        doc_instance.sents = mock_sents
        
        # For doc.__iter__ (used for word frequency calculation)
        if doc_tokens is None: # If specific doc tokens aren't provided, derive from sentences
            for sent_mock in mock_sents:
                 all_tokens_for_doc_iter.extend(list(sent_mock)) # Use the tokens from sentence mock
        else:
            all_tokens_for_doc_iter = doc_tokens

        doc_instance.__iter__.return_value = iter(all_tokens_for_doc_iter)
        return doc_instance
    return _factory


@pytest.fixture
def mock_spacy_nlp(mock_doc_factory):
    """Mocks the spaCy nlp object (the result of spacy.load())."""
    nlp = MagicMock(spec=spacy.language.Language)
    nlp.side_effect = lambda text: mock_doc_factory(text=text)
    nlp.Defaults = MagicMock()
    nlp.Defaults.stop_words = {"a", "is", "the", "and", "it", "of", "to", "in"}
    nlp.has_pipe = MagicMock(return_value=True) # Assume pipes exist for this main mock
    nlp.add_pipe = MagicMock()
    return nlp

@pytest.fixture
def mock_spacy_blank_nlp(mock_doc_factory):
    """Mocks a blank spaCy model for fallback testing."""
    nlp_blank = MagicMock(spec=spacy.language.Language)
    nlp_blank.side_effect = lambda text: mock_doc_factory(text=text) # Blank model still processes text
    nlp_blank.Defaults = MagicMock()
    nlp_blank.Defaults.stop_words = set() # Blank models have few/no stop words
    nlp_blank.has_pipe = MagicMock(return_value=False) # Simulate no pipes initially
    nlp_blank.add_pipe = MagicMock()
    return nlp_blank

# --- Test Cases ---

def test_agent_initialization_success(mock_spacy_nlp):
    """Test successful initialization with a mocked spaCy model."""
    with patch('spacy.load', return_value=mock_spacy_nlp) as mock_load:
        agent = ArgumentExtractionAgent(model_name='en_core_web_sm_test')
        mock_load.assert_called_once_with('en_core_web_sm_test')
        assert agent.nlp is mock_spacy_nlp

def test_agent_initialization_primary_fails_fallback_sm_succeeds(mock_spacy_nlp):
    """Test fallback to en_core_web_sm if the primary model fails."""
    def side_effect_spacy_load(model_name, **kwargs):
        if model_name == 'non_existent_model':
            raise OSError("Model 'non_existent_model' not found")
        elif model_name == "en_core_web_sm":
            return mock_spacy_nlp # Successfully load the fallback
        raise ValueError(f"Unexpected model load attempt: {model_name}")

    with patch('spacy.load', side_effect=side_effect_spacy_load) as mock_load, \
         patch('builtins.print') as mock_print:
        agent = ArgumentExtractionAgent(model_name='non_existent_model')
        
        expected_load_calls = [call('non_existent_model'), call('en_core_web_sm')]
        mock_load.assert_has_calls(expected_load_calls, any_order=False)
        mock_print.assert_any_call("spaCy model 'non_existent_model' not found. Please download it by running:")
        # The agent's __init__ also prints a warning if 'en_core_web_sm' is loaded as a fallback AND lacks vectors.
        # Our mock_spacy_nlp.has_pipe returns True, so this specific warning might not appear.
        # The crucial part is that it tries to load 'en_core_web_sm' and uses it.
        assert agent.nlp is mock_spacy_nlp

def test_agent_initialization_all_fallbacks_to_blank(mock_spacy_blank_nlp):
    """Test fallback to a blank model if specified and 'en_core_web_sm' fail."""
    def side_effect_spacy_load(model_name, **kwargs):
        if model_name == 'non_existent_model' or model_name == "en_core_web_sm":
            raise OSError(f"Model '{model_name}' not found")
        raise ValueError(f"Unexpected model load attempt: {model_name}")

    with patch('spacy.load', side_effect=side_effect_spacy_load) as mock_spacy_load_call, \
         patch('spacy.blank', return_value=mock_spacy_blank_nlp) as mock_spacy_blank_call, \
         patch('builtins.print') as mock_print:
        
        agent = ArgumentExtractionAgent(model_name='non_existent_model')
        
        expected_load_calls = [call('non_existent_model'), call('en_core_web_sm')]
        mock_spacy_load_call.assert_has_calls(expected_load_calls, any_order=False)
        mock_spacy_blank_call.assert_called_once_with("en")
        
        mock_print.assert_any_call("spaCy model 'non_existent_model' not found. Please download it by running:")
        mock_print.assert_any_call("Falling back to a blank English model ('en'). Processing quality will be affected.")
        assert agent.nlp is mock_spacy_blank_nlp
        mock_spacy_blank_nlp.add_pipe.assert_called_with("sentencizer")


def test_extract_arguments_valid_input(mock_spacy_nlp, mock_doc_factory):
    """Test extract_arguments with valid text inputs."""
    # Customize the mock_doc_factory for more controlled sentence/token generation if needed
    # For example, making sure some words are non-stop and alpha for summarization
    def custom_side_effect(text):
        if "base paper text" in text:
            return mock_doc_factory(text="This is the base paper text. It has important points. And more details.",
                                    sentences_texts=["This is the base paper text.", "It has important points.", "And more details."])
        elif "research angle text" in text:
            return mock_doc_factory(text="This is the research angle text. Focusing on specifics.",
                                    sentences_texts=["This is the research angle text.", "Focusing on specifics."])
        return mock_doc_factory(text=text) # Default

    mock_spacy_nlp.side_effect = custom_side_effect

    with patch('spacy.load', return_value=mock_spacy_nlp):
        agent = ArgumentExtractionAgent()
        base_text = "This is the base paper text. It has important points. And more details."
        angle_text = "This is the research angle text. Focusing on specifics."
        
        result = agent.extract_arguments(base_text, angle_text)
        
        assert "base_paper_summary" in result
        assert "research_angle_summary" in result
        assert isinstance(result["base_paper_summary"], str)
        assert isinstance(result["research_angle_summary"], str)
        
        assert len(result["base_paper_summary"]) > 0, "Base paper summary should not be empty."
        assert len(result["research_angle_summary"]) > 0, "Research angle summary should not be empty."
        # A more specific check based on how _extractive_summary works (top N sentences with most non-stop words)
        # With the mock, "important points" and "specifics" are likely candidates.
        assert "important points" in result["base_paper_summary"] or "base paper text" in result["base_paper_summary"]
        assert "specifics" in result["research_angle_summary"] or "research angle text" in result["research_angle_summary"]


def test_extract_arguments_empty_inputs(mock_spacy_nlp):
    """Test extract_arguments with empty string inputs."""
    with patch('spacy.load', return_value=mock_spacy_nlp):
        agent = ArgumentExtractionAgent()
        result = agent.extract_arguments("", "")
        expected_msg = "Could not generate summary due to lack of processable words."
        assert result["base_paper_summary"] == expected_msg
        assert result["research_angle_summary"] == expected_msg

def test_extract_arguments_very_short_inputs(mock_spacy_nlp):
    """Test extract_arguments with very short inputs."""
    with patch('spacy.load', return_value=mock_spacy_nlp):
        agent = ArgumentExtractionAgent()
        base_text = "Short."
        angle_text = "Angle."
        result = agent.extract_arguments(base_text, angle_text)
        assert result["base_paper_summary"] == "Short." # Summarizer should return the single sentence
        assert result["research_angle_summary"] == "Angle."


def test_extractive_summary_no_processable_words(mock_spacy_nlp, mock_token_factory):
    """Test _extractive_summary with text containing only stopwords or non-alpha."""
    # Mock a doc that consists only of stop words or non-alpha words
    stop_tokens = [mock_token_factory(text="the", is_stop=True), mock_token_factory(text="is", is_stop=True)]
    non_alpha_tokens = [mock_token_factory(text="123", is_alpha=False), mock_token_factory(text="!@#", is_alpha=False)]

    def custom_side_effect(text):
        doc = MagicMock(spec=spacy.tokens.Doc)
        if text == "the is":
            doc.sents = [MagicMock(text="the is", __iter__=lambda: iter(stop_tokens))]
            doc.__iter__.return_value = iter(stop_tokens)
        elif text == "123 !@#":
            doc.sents = [MagicMock(text="123 !@#", __iter__=lambda: iter(non_alpha_tokens))]
            doc.__iter__.return_value = iter(non_alpha_tokens)
        else: # Default behavior
            return mock_doc_factory()(text) # Call default doc factory
        return doc

    mock_spacy_nlp.side_effect = custom_side_effect
    
    with patch('spacy.load', return_value=mock_spacy_nlp):
        agent = ArgumentExtractionAgent()
        summary_stopwords = agent._extractive_summary("the is")
        assert summary_stopwords == "Could not generate summary due to lack of processable words."
        
        summary_non_alpha = agent._extractive_summary("123 !@#")
        assert summary_non_alpha == "Could not generate summary due to lack of processable words."


def test_extractive_summary_no_sentences(mock_spacy_nlp):
    """Test _extractive_summary when no sentences are detected."""
    def custom_side_effect(text):
        doc = MagicMock(spec=spacy.tokens.Doc)
        doc.sents = [] # No sentences
        doc.__iter__.return_value = iter([]) # No tokens
        return doc
        
    mock_spacy_nlp.side_effect = custom_side_effect
    with patch('spacy.load', return_value=mock_spacy_nlp):
        agent = ArgumentExtractionAgent()
        summary = agent._extractive_summary("   ") # Text that might yield no sentences
        assert summary == "Could not generate summary as no sentences were detected."

def test_extractive_summary_single_sentence_input(mock_spacy_nlp):
    """Test _extractive_summary with a single sentence input."""
    with patch('spacy.load', return_value=mock_spacy_nlp):
        agent = ArgumentExtractionAgent()
        text = "This is one complete sentence for summarization."
        summary = agent._extractive_summary(text, num_sentences=3) # Ask for more than available
        assert summary == text # Should return the sentence itself

def test_extractive_summary_ranking_logic(mock_spacy_nlp, mock_doc_factory, mock_token_factory):
    """Test that the summarizer prefers sentences with more scorable words."""
    
    # sentence1: "Common common word." (common appears twice)
    # sentence2: "Unique words are here." (unique, words, here - three scorable if not stop)
    
    tokens_s1_word1 = mock_token_factory(text="Common", lemma="common", is_stop=False, is_alpha=True)
    tokens_s1_word2 = mock_token_factory(text="common", lemma="common", is_stop=False, is_alpha=True)
    tokens_s1_word3 = mock_token_factory(text="word", lemma="word", is_stop=False, is_alpha=True)
    
    tokens_s2_word1 = mock_token_factory(text="Unique", lemma="unique", is_stop=False, is_alpha=True)
    tokens_s2_word2 = mock_token_factory(text="words", lemma="word", is_stop=False, is_alpha=True) # same lemma as s1
    tokens_s2_word3 = mock_token_factory(text="are", lemma="be", is_stop=True, is_alpha=True) # stop word
    tokens_s2_word4 = mock_token_factory(text="here", lemma="here", is_stop=False, is_alpha=True)

    doc_tokens = [
        tokens_s1_word1, tokens_s1_word2, tokens_s1_word3,
        tokens_s2_word1, tokens_s2_word2, tokens_s2_word3, tokens_s2_word4
    ]

    # Construct doc with specific sentences and tokens
    mock_doc_instance = mock_doc_factory(
        text="Custom text for ranking.", # This text isn't directly used by side_effect if sentences_texts is provided
        sentences_texts=["Common common word.", "Unique words are here."], # This will be used by side_effect
        doc_tokens=doc_tokens # This is important for frequency counting
    )
    # Override the mock_doc_factory's sentence tokenization for this specific test
    mock_doc_instance.sents[0].__iter__.return_value = iter([tokens_s1_word1, tokens_s1_word2, tokens_s1_word3])
    mock_doc_instance.sents[1].__iter__.return_value = iter([tokens_s2_word1, tokens_s2_word2, tokens_s2_word3, tokens_s2_word4])


    mock_spacy_nlp.side_effect = lambda text: mock_doc_instance
    mock_spacy_nlp.Defaults.stop_words = {"are"}


    with patch('spacy.load', return_value=mock_spacy_nlp):
        agent = ArgumentExtractionAgent()
        summary = agent._extractive_summary("Common common word. Unique words are here.", num_sentences=1)
        
        # "Common common word." -> common (2), word (1) -> score based on normalized freq
        # "Unique words are here." -> unique (1), word (1), here (1) (are is stop)
        # Word frequencies: common (2/2=1), word (2/2=1), unique(1/2=0.5), here(1/2=0.5)
        # Sentence 1 score: 1(common) + 1(common) + 1(word) = 3
        # Sentence 2 score: 0.5(unique) + 1(word) + 0.5(here) = 2
        # So, "Common common word." should be chosen.
        assert summary == "Common common word."

```
