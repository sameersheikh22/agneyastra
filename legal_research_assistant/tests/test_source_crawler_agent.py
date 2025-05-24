import pytest
from unittest.mock import patch, MagicMock, call
from legal_research_assistant.agents.source_crawler_agent import SourceCrawlerAgent

# --- Fixtures ---

@pytest.fixture
def mock_tavily_client_fixture_factory():
    """Factory to create a mock TavilyClient with customizable search responses."""
    def _factory(search_responses=None): # search_responses can be a list of dicts or a single dict
        client = MagicMock(spec=SourceCrawlerAgent) # Mocking TavilyClient itself
        
        if search_responses is None: # Default successful response
            default_ssrn_response = {
                "results": [
                    {"title": "SSRN Paper 1", "url": "https://papers.ssrn.com/id=1", "content": "Abstract SSRN 1", "score": 0.9},
                    {"title": "SSRN Paper 2", "url": "https://ssrn.com/id=2", "content": "Abstract SSRN 2", "score": 0.8},
                    {"title": "Non-SSRN Paper", "url": "https://example.com/paper", "content": "Abstract Other", "score": 0.85},
                ]
            }
            client.search = MagicMock(return_value=default_ssrn_response)
        elif isinstance(search_responses, list): # If multiple responses for sequential calls
            client.search = MagicMock(side_effect=search_responses)
        elif isinstance(search_responses, dict) or search_responses is None: # Single response or error
             client.search = MagicMock(return_value=search_responses)
        elif isinstance(search_responses, Exception): # Simulate an API error
            client.search = MagicMock(side_effect=search_responses)
            
        return client
    return _factory


# --- Test Cases for Initialization ---

def test_sca_initialization_success():
    agent = SourceCrawlerAgent(api_key="fake_tavily_key")
    assert agent.tavily_client is not None # TavilyClient instance is created
    # We can't easily assert agent.api_key as it's not stored directly after client creation

def test_sca_initialization_missing_key():
    with patch('builtins.print') as mock_print:
        agent = SourceCrawlerAgent(api_key=None)
        assert agent.tavily_client is None
        mock_print.assert_any_call("Warning: Tavily API key is missing or a placeholder. SourceCrawlerAgent may not function.")

def test_sca_initialization_placeholder_key():
    with patch('builtins.print') as mock_print:
        agent = SourceCrawlerAgent(api_key="YOUR_TAVILY_KEY_HERE")
        assert agent.tavily_client is None
        mock_print.assert_any_call("Warning: Tavily API key is missing or a placeholder. SourceCrawlerAgent may not function.")

# --- Test Cases for search_ssrn ---

@patch('legal_research_assistant.agents.source_crawler_agent.TavilyClient')
def test_search_ssrn_success(MockTavilyClientConstructor, mock_tavily_client_fixture_factory):
    # Prepare a mock client with a specific SSRN response
    ssrn_response = {
        "results": [
            {"title": "SSRN Paper 1", "url": "https://papers.ssrn.com/id=1", "content": "Abstract SSRN 1", "score": 0.9},
            {"title": "SSRN Paper 2 (root)", "url": "https://ssrn.com/id=2", "content": "Abstract SSRN 2", "score": 0.8},
            {"title": "Non-SSRN Paper", "url": "https://example.com/paper", "content": "Abstract Other", "score": 0.85},
        ]
    }
    mock_client_instance = mock_tavily_client_fixture_factory(search_responses=ssrn_response)
    MockTavilyClientConstructor.return_value = mock_client_instance
    
    agent = SourceCrawlerAgent(api_key="fake_key")
    keywords = ["legal tech"]
    results = agent.search_ssrn(keywords, max_results_per_keyword=5)
    
    assert len(results) == 2 # Non-SSRN paper should be filtered
    assert results[0]['title'] == "SSRN Paper 1"
    assert results[0]['source'] == "SSRN (via Tavily)"
    assert "ssrn.com" in results[0]['url']
    assert results[1]['title'] == "SSRN Paper 2 (root)"
    assert "ssrn.com" in results[1]['url']
    
    mock_client_instance.search.assert_called_once_with(
        query="legal tech site:papers.ssrn.com OR site:ssrn.com",
        search_depth="basic", 
        max_results=5
    )

@patch('legal_research_assistant.agents.source_crawler_agent.TavilyClient')
def test_search_ssrn_api_error(MockTavilyClientConstructor, mock_tavily_client_fixture_factory):
    mock_client_instance = mock_tavily_client_fixture_factory(search_responses=Exception("Tavily API Error"))
    MockTavilyClientConstructor.return_value = mock_client_instance
    
    agent = SourceCrawlerAgent(api_key="fake_key")
    with patch('builtins.print') as mock_print:
        results = agent.search_ssrn(["oops"], max_results_per_keyword=1)
        assert results == []
        mock_print.assert_any_call("   - Error searching with Tavily for keyword 'oops': Tavily API Error")

@patch('legal_research_assistant.agents.source_crawler_agent.TavilyClient')
def test_search_ssrn_no_results(MockTavilyClientConstructor, mock_tavily_client_fixture_factory):
    mock_client_instance = mock_tavily_client_fixture_factory(search_responses={"results": []})
    MockTavilyClientConstructor.return_value = mock_client_instance
    
    agent = SourceCrawlerAgent(api_key="fake_key")
    results = agent.search_ssrn(["findnothing"], max_results_per_keyword=3)
    assert results == []

def test_search_ssrn_uninitialized_client():
    with patch('builtins.print') as mock_print: # Capture initialization warning
        agent = SourceCrawlerAgent(api_key=None) 
    results = agent.search_ssrn(["test"], max_results_per_keyword=1)
    assert results == []
    mock_print.assert_any_call("Tavily client not initialized due to missing API key. Cannot search.")


@patch('legal_research_assistant.agents.source_crawler_agent.TavilyClient')
def test_search_ssrn_deduplication(MockTavilyClientConstructor, mock_tavily_client_fixture_factory):
    response1 = {"results": [
        {"title": "Paper A", "url": "https://papers.ssrn.com/id=1", "content": "Abstract A", "score": 0.9},
        {"title": "Paper B", "url": "https://papers.ssrn.com/id=2", "content": "Abstract B", "score": 0.8},
    ]}
    response2 = {"results": [ # Second keyword returns one old and one new
        {"title": "Paper A Duplicate", "url": "https://papers.ssrn.com/id=1", "content": "Abstract A again", "score": 0.91},
        {"title": "Paper C", "url": "https://papers.ssrn.com/id=3", "content": "Abstract C", "score": 0.85},
    ]}
    mock_client_instance = mock_tavily_client_fixture_factory(search_responses=[response1, response2])
    MockTavilyClientConstructor.return_value = mock_client_instance
    
    agent = SourceCrawlerAgent(api_key="fake_key")
    keywords = ["keyword1", "keyword2"]
    results = agent.search_ssrn(keywords, max_results_per_keyword=2)
    
    assert len(results) == 3 # Paper A (first instance), Paper B, Paper C
    urls = [r['url'] for r in results]
    assert "https://papers.ssrn.com/id=1" in urls
    assert "https://papers.ssrn.com/id=2" in urls
    assert "https://papers.ssrn.com/id=3" in urls
    
    # Ensure the first encountered version of Paper A is kept (based on title)
    paper_a_results = [r for r in results if r['url'] == "https://papers.ssrn.com/id=1"]
    assert len(paper_a_results) == 1
    assert paper_a_results[0]['title'] == "Paper A" 
    
    assert mock_client_instance.search.call_count == 2


# --- Test Cases for search_jstor ---

@patch('legal_research_assistant.agents.source_crawler_agent.TavilyClient')
def test_search_jstor_success(MockTavilyClientConstructor, mock_tavily_client_fixture_factory):
    jstor_response = {
        "results": [
            {"title": "JSTOR Article 1", "url": "https://www.jstor.org/stable/1001", "content": "Abstract JSTOR 1", "score": 0.92},
            {"title": "JSTOR Article 2", "url": "http://jstor.org/stable/1002", "content": "Abstract JSTOR 2", "score": 0.88}, # Note http
            {"title": "Non-JSTOR Article", "url": "https://example.com/article", "content": "Abstract Other", "score": 0.80},
        ]
    }
    mock_client_instance = mock_tavily_client_fixture_factory(search_responses=jstor_response)
    MockTavilyClientConstructor.return_value = mock_client_instance

    agent = SourceCrawlerAgent(api_key="fake_key")
    keywords = ["history"]
    results = agent.search_jstor(keywords, max_results_per_keyword=5)

    assert len(results) == 2 # Non-JSTOR article should be filtered
    assert results[0]['title'] == "JSTOR Article 1"
    assert results[0]['source'] == "JSTOR (via Tavily)"
    assert "jstor.org" in results[0]['url']
    assert "jstor.org" in results[1]['url']
    
    mock_client_instance.search.assert_called_once_with(
        query="history site:jstor.org",
        search_depth="basic", 
        max_results=5
    )

@patch('legal_research_assistant.agents.source_crawler_agent.TavilyClient')
def test_search_jstor_api_error(MockTavilyClientConstructor, mock_tavily_client_fixture_factory):
    mock_client_instance = mock_tavily_client_fixture_factory(search_responses=Exception("Tavily JSTOR Error"))
    MockTavilyClientConstructor.return_value = mock_client_instance
    
    agent = SourceCrawlerAgent(api_key="fake_key")
    with patch('builtins.print') as mock_print:
        results = agent.search_jstor(["failure"], max_results_per_keyword=1)
        assert results == []
        mock_print.assert_any_call("   - Error searching JSTOR with Tavily for keyword 'failure': Tavily JSTOR Error")

def test_search_jstor_uninitialized_client():
    with patch('builtins.print') as mock_print: # Capture initialization warning
        agent = SourceCrawlerAgent(api_key=None) 
    results = agent.search_jstor(["test"], max_results_per_keyword=1)
    assert results == []
    mock_print.assert_any_call("Tavily client not initialized due to missing API key. Cannot search JSTOR.")

# Test for old search_sources method (ensure it calls search_ssrn and prints warning)
@patch('legal_research_assistant.agents.source_crawler_agent.TavilyClient')
def test_search_sources_deprecated_calls_ssrn(MockTavilyClientConstructor, mock_tavily_client_fixture_factory):
    ssrn_response = {"results": [{"title": "Test SSRN", "url": "https://papers.ssrn.com/id=test", "content": "Abstract", "score": 0.5}]}
    mock_client_instance = mock_tavily_client_fixture_factory(search_responses=ssrn_response)
    MockTavilyClientConstructor.return_value = mock_client_instance

    agent = SourceCrawlerAgent(api_key="fake_key")
    with patch.object(agent, 'search_ssrn', wraps=agent.search_ssrn) as mock_search_ssrn_method, \
         patch('builtins.print') as mock_print:
        
        agent.search_sources(["deprecated_test"])
        
        mock_search_ssrn_method.assert_called_once_with(["deprecated_test"])
        mock_print.assert_any_call("Warning: `search_sources` is deprecated. It currently only calls `search_ssrn`.")
        mock_print.assert_any_call("         Consider calling `search_ssrn` and `search_jstor` methods directly from main.")

```
