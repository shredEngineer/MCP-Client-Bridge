#  Copyright © 2025 Dr.-Ing. Paul Wilhelm <paul@wilhelm.dev>
#  This file is part of Archive Agent. See LICENSE for details.

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client


server_url = "http://192.168.178.39:8008/sse"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _maybe_json(text: str) -> Union[Dict[str, Any], List[Any], str]:
    """
    Try to parse JSON text; if parsing fails, return the original string.

    :param text: Input string.
    :return: Parsed JSON object, or original string if not JSON.
    """
    try:
        return json.loads(text)
    except Exception:
        return text


class McpClient:
    """
    MCP client.
    """

    def __init__(self) -> None:
        """
        Initialize MCP client.
        """
        self.session: Optional[ClientSession] = None
        self._session_context: Optional[ClientSession] = None  # context manager proxy, not the logical session
        self._streams_context = None  # sse_client() async context manager

    async def connect_to_sse_server(self) -> None:
        """
        Connect to MCP server running with SSE transport.

        Raises
        ------
        RuntimeError
            If the connection or initialization fails.
        """
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session = await self._session_context.__aenter__()
        await self.session.initialize()

        # List available tools to verify connection
        response = await self.session.list_tools()
        tools = response.tools
        logger.info("Initialized SSE client. Tools: %s", [tool.name for tool in tools])

    async def cleanup(self) -> None:
        """
        Clean up the session and streams.
        """
        try:
            if self._session_context is not None:
                await self._session_context.__aexit__(None, None, None)
        finally:
            if self._streams_context is not None:
                await self._streams_context.__aexit__(None, None, None)

    async def _call_tool_json(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool and return a best-effort decoded JSON payload.

        The MCP spec transports rich content as a list. We accept:
        - a direct JSON object (some servers serialize dicts directly),
        - a single text part containing JSON,
        - otherwise we return the raw content structure.

        Parameters
        ----------
        name:
            Tool name, e.g. "get_search_result".
        arguments:
            JSON-serializable dict of arguments.

        Returns
        -------
        Any
            Decoded JSON (dict/list) or raw content.
        """
        assert self.session is not None
        resp = await self.session.call_tool(name=name, arguments=arguments)

        # Common patterns:
        # 1) resp.content is a list of parts; each part may have `.text`
        # 2) some implementations put the object directly into `.content[0].text` (JSON)
        # 3) some return already structured content
        content = getattr(resp, "content", None)

        if isinstance(content, list) and content:
            part = content[0]
            text = getattr(part, "text", None)
            if isinstance(text, str):
                return _maybe_json(text)
            # Some SDKs place the dict under `.data` or similar; try common attributes:
            data = getattr(part, "data", None)
            if data is not None:
                return data
            return content

        # Fallback: try direct attribute
        data = getattr(resp, "result", None)
        if data is not None:
            return data

        return content

    async def ask(self, question: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Query the MCP server for a question using both tools concurrently.

        This runs:
        - get_search_result(question): returns {file_path: score, ...}
        - get_answer_rag(question): returns the structured RAG answer schema

        Parameters
        ----------
        question:
            Natural-language query.

        Returns
        -------
        (search_map, rag_result)
            *search_map* is a mapping of file_path → relevance score.
            *rag_result* is the RAG answer schema returned by the server.
        """
        search_coro = self._call_tool_json("get_search_result", {"question": question})
        rag_coro = self._call_tool_json("get_answer_rag", {"question": question})

        search_result_raw, rag_result_raw = await asyncio.gather(search_coro, rag_coro)

        # Normalize search map
        if isinstance(search_result_raw, dict):
            search_map: Dict[str, float] = {str(k): float(v) for k, v in search_result_raw.items()}
        else:
            logger.warning("Unexpected search result shape: %r", type(search_result_raw))
            search_map = {}

        # RAG result should be a dict with known keys; if not, pass through raw
        rag_result: Dict[str, Any]
        if isinstance(rag_result_raw, dict):
            rag_result = rag_result_raw
        else:
            logger.warning("Unexpected RAG result shape: %r", type(rag_result_raw))
            rag_result = {"raw": rag_result_raw}

        return search_map, rag_result


async def async_main(question: Optional[str] = None) -> None:
    """
    Main entry point for MCP client.

    Parameters
    ----------
    question:
        Optional question to ask. If None, read from stdin.
    """
    if not question:
        try:
            question = input("🧠 Ask Archive Agent… ").strip()
        except EOFError:
            question = None

    if not question:
        logger.error("No question provided.")
        return

    client = McpClient()
    try:
        await client.connect_to_sse_server()

        search_map, rag = await client.ask(question)

        # Pretty print summary
        logger.info("Top relevant files:")
        for path, score in sorted(search_map.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            logger.info("  %.4f  %s", score, path)

        if isinstance(rag, dict):
            logger.info("Answer (rephrased): %s", rag.get("question_rephrased"))
            logger.info("Conclusion: %s", rag.get("answer_conclusion"))

            answers = rag.get("answer_list") or []
            if isinstance(answers, list):
                for i, item in enumerate(answers, 1):
                    ans = item.get("answer")
                    refs = item.get("chunk_ref_list")
                    logger.info("— Part %d:", i)
                    if isinstance(ans, str):
                        logger.info("  %s", ans)
                    if isinstance(refs, list):
                        logger.info("  refs: %s", refs)

            rejected = rag.get("is_rejected")
            if rejected:
                logger.warning("RAG rejected: %s", rag.get("rejection_reason"))

        else:
            logger.info("RAG (raw): %r", rag)

    finally:
        await client.cleanup()


def main() -> None:
    """
    Synchronous entry point.
    """
    import sys
    q: Optional[str] = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(async_main(q))


if __name__ == "__main__":
    main()
