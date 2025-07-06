"""
Unified request handler for LLM providers.

This module provides a standardized way to make API requests to different LLM providers,
handling common patterns like authentication, rate limiting, retries, and error handling.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple, Union

import aiohttp
from ..providers.base import AuthenticationError, ProviderError, RateLimitError

logger = logging.getLogger(__name__)


class RequestHandler:
    """Unified request handler for LLM API requests"""

    def __init__(
        self,
        api_key: Optional[str],
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._retry_count = 0

    async def make_request(
        self,
        endpoint: str,
        headers: Dict[str, str],
        data: Dict[str, Any],
        response_key_path: Tuple[str, ...],
    ) -> str:
        """
        Make an API request with standardized error handling and retries.

        Args:
            endpoint: API endpoint path (will be appended to base_url)
            headers: HTTP headers for the request
            data: Request payload
            response_key_path: Tuple of keys to navigate to extract the response text
                               (e.g., ("choices", 0, "message", "content") for OpenAI)

        Returns:
            Extracted response text from the API
        """
        last_exception = None
        self._retry_count = 0

        while self._retry_count < self.max_retries:
            try:
                async with aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(limit=30),
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as session:
                    async with session.post(
                        f"{self.base_url}/{endpoint.lstrip('/')}",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        if response.status == 401:
                            raise AuthenticationError("Invalid API key")
                        elif response.status == 429:
                            retry_after = float(
                                response.headers.get("Retry-After", self.retry_delay)
                            )
                            # Instead of raising immediately, we'll handle the retry
                            self._retry_count += 1
                            if self._retry_count >= self.max_retries:
                                raise RateLimitError(retry_after=retry_after)
                            await asyncio.sleep(retry_after)
                            continue
                        elif response.status != 200:
                            error_text = await response.text()
                            raise ProviderError(
                                f"API error (status {response.status}): {error_text}"
                            )

                        response_data = await response.json()
                        return self._extract_response(response_data, response_key_path)

            except RateLimitError as e:
                last_exception = e
                raise  # If we've exceeded max retries, propagate the error
            except aiohttp.ClientError as e:
                last_exception = e
                self._retry_count += 1
                if self._retry_count >= self.max_retries:
                    raise ProviderError(
                        f"Network error after {self._retry_count} retries: {str(e)}"
                    )
                await asyncio.sleep(self.retry_delay * self._retry_count)
            except (KeyError, IndexError, ValueError) as e:
                raise ProviderError(f"Invalid API response: {str(e)}")

        raise last_exception or ProviderError("Maximum retry attempts exceeded")

    def _extract_response(
        self, response_data: Dict[str, Any], key_path: Tuple[str, ...]
    ) -> str:
        """
        Extract the response text from a nested dictionary using a key path.

        Args:
            response_data: Response data dictionary
            key_path: Tuple of keys to navigate to extract the response text

        Returns:
            Extracted response text
        """
        result = response_data
        for key in key_path:
            if isinstance(key, int):
                if not isinstance(result, list) or len(result) <= key:
                    raise ProviderError(
                        f"Invalid response structure: expected list at {key_path}"
                    )
                result = result[key]
            else:
                if not isinstance(result, dict) or key not in result:
                    raise ProviderError(
                        f"Invalid response structure: missing key '{key}' in {key_path}"
                    )
                result = result[key]

        if not isinstance(result, str):
            raise ProviderError(
                f"Invalid response type: expected string at {key_path}, got {type(result)}"
            )

        return result