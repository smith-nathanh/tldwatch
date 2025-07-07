from tldwatch.utils.url_parser import extract_video_id, is_youtube_url


def test_youtube_url_validation():
    """Test various YouTube URL formats"""
    valid_urls = [
        "https://www.youtube.com/watch?v=abcdefghijk",
        "https://youtu.be/abcdefghijk",
        "https://youtube.com/watch?v=abcdefghijk&feature=shared",
        "https://m.youtube.com/watch?v=abcdefghijk",
    ]
    invalid_urls = [
        "https://www.notyoutube.com/watch?v=abcdefghijk",
        "https://www.youtube.com/watch",
        "https://youtu.be/",
        "https://youtube.com/",
        "https://m.youtube.com/",
    ]

    for url in valid_urls:
        assert is_youtube_url(url) is True

    for url in invalid_urls:
        assert is_youtube_url(url) is False


def test_video_id_extraction():
    """Test extracting video IDs from different URL formats"""
    test_cases = [
        ("https://www.youtube.com/watch?v=abcdefghijk", "abcdefghijk"),
        ("https://youtu.be/abcdefghijk", "abcdefghijk"),
        ("https://youtube.com/watch?v=abcdefghijk&feature=shared", "abcdefghijk"),
        ("https://m.youtube.com/watch?v=abcdefghijk", "abcdefghijk"),
        ("https://www.youtube.com/watch?v=abcdefghijk&list=PLAYLIST_ID", "abcdefghijk"),
        ("https://www.youtube.com/watch?v=abcdefghijk&start=10", "abcdefghijk"),
    ]

    for url, expected_video_id in test_cases:
        assert extract_video_id(url) == expected_video_id


def test_invalid_urls():
    """Test handling of invalid URLs"""
    invalid_urls = [
        "https://www.notyoutube.com/watch?v=abcdefghijk",
        "https://www.youtube.com/watch",
        "https://youtu.be/",
        "https://youtube.com/",
        "https://m.youtube.com/",
        "https://www.youtube.com/watch?v=",
        "https://www.youtube.com/watch?v=invalid_video_id",
    ]

    for url in invalid_urls:
        assert extract_video_id(url) is None
