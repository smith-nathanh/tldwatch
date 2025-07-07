# TLDWatch Test Suite

This directory contains comprehensive unit and integration tests for the tldwatch library.

## Test Structure

```
tests/
├── conftest.py              # Shared pytest fixtures and configuration
├── unit/                    # Unit tests for individual components
│   ├── cli/
│   │   └── test_main.py    # CLI functionality tests
│   ├── core/
│   │   ├── test_summarizer.py          # Main summarizer tests
│   │   ├── test_unified_provider.py    # Provider system tests
│   │   ├── test_user_config.py         # User configuration tests
│   │   └── test_proxy_config.py        # Proxy configuration tests
│   └── utils/
│       ├── test_cache.py               # Cache functionality tests
│       └── test_url_parser.py          # URL parsing tests
└── integration/             # Integration tests
    ├── test_summarizer_integration.py  # End-to-end summarization tests
    └── test_cli_integration.py         # CLI workflow tests
```

## Running Tests

### Prerequisites

Install the development dependencies:

```bash
# Using uv (recommended)
uv add --group dev pytest pytest-asyncio pytest-mock pytest-cov

# Or using pip
pip install pytest pytest-asyncio pytest-mock pytest-cov
```

### Running All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tldwatch --cov-report=html

# Run with verbose output
pytest -v
```

### Running Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/core/test_summarizer.py

# Run specific test class
pytest tests/unit/core/test_summarizer.py::TestSummarizer

# Run specific test method
pytest tests/unit/core/test_summarizer.py::TestSummarizer::test_summarize_direct_text
```

### Running Tests with Different Filters

```bash
# Run tests matching a pattern
pytest -k "cache"

# Run tests with specific markers (if you add markers)
pytest -m "slow"

# Skip integration tests (faster for development)
pytest tests/unit/
```

## Test Coverage

The test suite covers:

### Core Functionality (Unit Tests)
- **Summarizer**: Complete summarization workflow with various inputs
- **UnifiedProvider**: Provider initialization, configuration, and API calls
- **Cache**: Summary and transcript caching with different scenarios
- **UserConfig**: Configuration loading, validation, and defaults
- **ProxyConfig**: Proxy creation and configuration
- **URLParser**: YouTube URL validation and video ID extraction

### CLI Functionality (Unit Tests)
- **Main CLI**: Argument parsing, command execution, error handling
- **Configuration Commands**: Creating and showing configuration
- **Proxy Configuration**: Command-line proxy setup
- **Output Options**: File output, verbose logging, cache control

### Integration Tests
- **End-to-End Summarization**: Complete workflow from input to cached output
- **CLI Workflows**: Real CLI command execution with mocked external services
- **Cache Integration**: Transcript and summary caching across multiple runs
- **Provider Switching**: Different LLM providers with same video

## Test Features

### Mocking Strategy
- **External APIs**: YouTube Transcript API and LLM provider APIs are mocked
- **File System**: Temporary directories for cache and config testing
- **Environment Variables**: Controlled environment for API keys and settings
- **Network Requests**: aiohttp sessions are mocked for provider API calls

### Fixtures Available
- `temp_cache_dir`: Temporary cache directory
- `temp_config_dir`: Temporary config directory  
- `mock_youtube_transcript`: Pre-configured YouTube transcript data
- `mock_aiohttp_session`: Mocked HTTP session for API calls
- `sample_video_id`: Consistent test video ID
- `sample_video_url`: Consistent test video URL
- `sample_transcript_text`: Sample transcript content
- `mock_env_vars`: API keys for testing
- `sample_user_config`: Complete user configuration

### Best Practices Followed
- **Isolation**: Each test is independent with clean state
- **Comprehensive Coverage**: Testing happy path, edge cases, and error conditions
- **Real Scenarios**: Integration tests simulate actual usage patterns
- **Performance**: Unit tests are fast; integration tests validate complete workflows
- **Maintainability**: Clear test names and good documentation

## Key Test Scenarios

### Cache Testing
- Summary caching with different provider configurations
- Transcript caching and reuse
- Cache hit/miss scenarios
- Cache clearing and forced regeneration
- Cache statistics and management

### Provider Testing
- Multiple LLM providers (OpenAI, Anthropic, etc.)
- Different models and configuration options
- Chunking strategies for long content
- Error handling for API failures
- Rate limiting and retry logic

### CLI Testing
- Various input types (URLs, video IDs, direct text)
- Command-line option combinations
- Configuration file creation and management
- Output formatting and file writing
- Error handling and user feedback

### Integration Testing
- Complete video summarization workflow
- Cache persistence across runs
- Provider switching with same content
- CLI commands with real component interaction
- Error propagation and recovery

## Adding New Tests

When adding new functionality to tldwatch:

1. **Add Unit Tests**: Test the new component in isolation
2. **Update Integration Tests**: Ensure new features work in complete workflows
3. **Update Fixtures**: Add new test data or mocks as needed
4. **Document Coverage**: Update this README with new test scenarios

### Example Test Pattern

```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_feature_success(self, mock_dependency):
        """Test successful operation."""
        # Arrange
        setup_test_data()
        
        # Act
        result = new_feature()
        
        # Assert
        assert result == expected_value
    
    def test_feature_error_handling(self):
        """Test error handling."""
        with pytest.raises(ExpectedError):
            new_feature(invalid_input)
```

## Debugging Tests

```bash
# Run with detailed output
pytest -v -s

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Run specific failing test
pytest tests/unit/core/test_summarizer.py::TestSummarizer::test_failing_method -v
```

## Continuous Integration

These tests are designed to run in CI environments:
- No external API dependencies (everything mocked)
- Temporary file cleanup
- Consistent test data
- Fast execution for development feedback

The test suite should pass completely before merging any changes.
