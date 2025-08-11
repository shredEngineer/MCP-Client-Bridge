#  Copyright © 2025 Dr.-Ing. Paul Wilhelm <paul@wilhelm.dev>
#  This file is part of Archive Agent. See LICENSE for details.

import asyncio
import logging
from typing import Optional, Dict, Any, List, Union

from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from rich import print_json


server_url = "http://192.168.178.39:8008/sse"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class McpClient:
    """
    MCP client.
    """

    def __init__(self) -> None:
        """
        Initialize MCP client.
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._session_context = None
        self._streams_context = None

    async def connect_to_sse_server(self) -> None:
        """
        Connect to MCP server running with SSE transport.
        """
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session = await self._session_context.__aenter__()

        await self.session.initialize()

        # List available tools to verify connection
        logger.info("Initialized SSE client...")
        logger.info("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        logger.info("Connected to server with tools: %s", [tool.name for tool in tools])

    async def cleanup(self) -> None:
        """
        Clean up the session and streams.
        """
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    @staticmethod
    def _as_json_dict(payload: Any) -> Dict[str, Any]:
        """
        Convert a tool call result into a JSON-like dict.

        Notes
        -----
        We try a few common MCP content shapes to keep this resilient:
        - If the payload is already a dict, return it.
        - If it has ``content`` items with ``.json`` fields, use the first one.
        - If it has text that looks like JSON, attempt to parse it.
        """
        if isinstance(payload, dict):
            return payload

        # Tool results from mcp can have a `.content` list with parts.
        content: Optional[List[Any]] = getattr(payload, "content", None)
        if content:
            part0 = content[0]
            # Prefer structured JSON if present
            json_data = getattr(part0, "json", None)
            if isinstance(json_data, dict):
                return json_data  # type: ignore[return-value]

            # Fall back to text that may contain JSON
            text_data = getattr(part0, "text", None)
            if isinstance(text_data, str):
                import json as _json
                try:
                    return _json.loads(text_data)
                except Exception:
                    pass  # not JSON; continue

        # Last resort: try model_dump if it exists (pydantic-style)
        dump_fn = getattr(payload, "model_dump", None)
        if callable(dump_fn):
            out = dump_fn()
            if isinstance(out, dict):
                return out

        # Give up, wrap as-is
        return {"result": payload}

    async def get_search_results(self, question: str) -> Dict[str, float]:
        """
        Call the ``get_search_result`` MCP tool.

        Parameters
        ----------
        question:
            Natural-language query.

        Returns
        -------
        Dict[str, float]
            Mapping ``{file_path: relevance_score}``.
        """
        assert self.session is not None
        result = await self.session.call_tool("get_search_result", arguments={"question": question})
        data = self._as_json_dict(result)
        return {str(k): float(v) for k, v in data.items()}

    async def get_answer_rag(self, question: str) -> Dict[str, Any]:
        """
        Call the ``get_answer_rag`` MCP tool.

        Parameters
        ----------
        question:
            Natural-language question.

        Returns
        -------
        Dict[str, Any]
            RAG answer payload as returned by the server.
        """
        assert self.session is not None
        result = await self.session.call_tool("get_answer_rag", arguments={"question": question})
        return self._as_json_dict(result)


async def async_main() -> None:
    """
    Main entry point for MCP client.
    """
    client = McpClient()
    try:
        await client.connect_to_sse_server()

        question: str = "Which files mention donuts?"

        # Search results
        search_map: Dict[str, float] = await client.get_search_results(question)
        print_json(data={"search_results": search_map})

        # Full RAG answer
        rag_answer: Dict[str, Any] = await client.get_answer_rag(question)
        print_json(data={"answer_rag": rag_answer})

    finally:
        await client.cleanup()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
