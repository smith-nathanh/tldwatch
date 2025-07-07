"""
Unit tests for URL parser utility.
Tests URL validation and video ID extraction functionality.
"""

from tldwatch.utils.url_parser import extract_video_id, is_youtube_url


class TestIsYouTubeUrl:
    """Test YouTube URL validation."""

    def test_valid_youtube_urls(self):
        """Test valid YouTube URL formats."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "http://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "http://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=123s",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        ]

        for url in valid_urls:
            assert is_youtube_url(url), f"Failed for URL: {url}"

    def test_invalid_youtube_urls(self):
        """Test invalid URL formats."""
        invalid_urls = [
            "https://www.google.com",
            "https://vimeo.com/123456789",
            "not_a_url",
            "",
            None,
            "dQw4w9WgXcQ",  # Just video ID, not a URL
            "https://www.youtube.com/",  # No video ID
            "https://www.youtube.com/channel/UC123456789",
            "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi",
        ]

        for url in invalid_urls:
            assert not is_youtube_url(url), f"Incorrectly validated URL: {url}"


class TestExtractVideoId:
    """Test video ID extraction from URLs."""

    def test_extract_from_watch_urls(self):
        """Test extraction from standard watch URLs."""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ]

        for url, expected_id in test_cases:
            result = extract_video_id(url)
            assert (
                result == expected_id
            ), f"Failed for URL: {url}, got {result}, expected {expected_id}"

    def test_extract_from_shortened_urls(self):
        """Test extraction from shortened youtu.be URLs."""
        test_cases = [
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ?t=123", "dQw4w9WgXcQ"),
        ]

        for url, expected_id in test_cases:
            result = extract_video_id(url)
            assert (
                result == expected_id
            ), f"Failed for URL: {url}, got {result}, expected {expected_id}"

    def test_extract_with_parameters(self):
        """Test extraction from URLs with additional parameters."""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=123s", "dQw4w9WgXcQ"),
            (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi",
                "dQw4w9WgXcQ",
            ),
            (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=youtu.be",
                "dQw4w9WgXcQ",
            ),
            (
                "https://www.youtube.com/watch?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&v=dQw4w9WgXcQ",
                "dQw4w9WgXcQ",
            ),
        ]

        for url, expected_id in test_cases:
            result = extract_video_id(url)
            assert (
                result == expected_id
            ), f"Failed for URL: {url}, got {result}, expected {expected_id}"

    def test_extract_from_invalid_urls(self):
        """Test extraction from invalid URLs returns None."""
        invalid_urls = [
            "https://www.google.com",
            "https://vimeo.com/123456789",
            "not_a_url",
            "",
            None,
            "https://www.youtube.com/",  # No video ID
            "https://www.youtube.com/channel/UC123456789",
            "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi",
        ]

        for url in invalid_urls:
            result = extract_video_id(url)
            assert (
                result is None
            ), f"Should return None for invalid URL: {url}, but got {result}"

    def test_extract_various_video_id_formats(self):
        """Test extraction with various valid video ID formats."""
        # Test different valid video ID patterns
        test_cases = [
            ("https://www.youtube.com/watch?v=1234567890A", "1234567890A"),
            ("https://www.youtube.com/watch?v=aBcDeFgHiJk", "aBcDeFgHiJk"),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/7Sse6P5xbEc", "7Sse6P5xbEc"),
            ("https://youtu.be/MueCRSZ3RQ0", "MueCRSZ3RQ0"),
        ]

        for url, expected_id in test_cases:
            result = extract_video_id(url)
            assert (
                result == expected_id
            ), f"Failed for URL: {url}, got {result}, expected {expected_id}"

    def test_extract_edge_cases(self):
        """Test edge cases for video ID extraction."""
        # Test URLs with malformed or unusual patterns
        edge_cases = [
            ("https://www.youtube.com/watch?v=", None),  # Empty video ID
            (
                "https://www.youtube.com/watch?video=dQw4w9WgXcQ",
                None,
            ),  # Wrong parameter name
            ("https://youtu.be/", None),  # No video ID in shortened URL
            ("https://www.youtube.com/watch", None),  # No parameters at all
        ]

        for url, expected_result in edge_cases:
            result = extract_video_id(url)
            assert (
                result == expected_result
            ), f"Failed for edge case: {url}, got {result}, expected {expected_result}"
