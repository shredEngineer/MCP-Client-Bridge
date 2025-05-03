#  Copyright © 2025 Dr.-Ing. Paul Wilhelm <paul@wilhelm.dev>
#  This file is part of Archive Agent. See LICENSE for details.

import asyncio
import logging
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client


server_url = "http://localhost:8008/sse"

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


async def main() -> None:
    """
    Main entry point for MCP client.
    """
    client = McpClient()
    try:
        await client.connect_to_sse_server()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
