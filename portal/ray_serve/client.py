"""HTTP client for communicating with the internal Ray Serve cluster."""

import logging
from typing import Any

import httpx

from portal.config import settings

logger = logging.getLogger(__name__)


class RayClient:
    """
    A generalized client to talk to any Ray Serve ML Deployment.
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.ray_serve_base
        # We use an async client so the web server doesn't freeze waiting for ML
        self.http_client = httpx.AsyncClient(timeout=60.0)

    async def predict(self, endpoint_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Generic method to send JSON data to ANY Ray deployment via HTTP.
        """
        url = f"{self.base_url}/{endpoint_name}"

        try:
            response = await self.http_client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

        except httpx.RequestError as e:
            logger.error(f"Failed to communicate with Ray deployment {endpoint_name}: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Ray deployment {endpoint_name} returned error status: {e.response.status_code} - {e.response.text}"
            )
            raise

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.http_client.aclose()
