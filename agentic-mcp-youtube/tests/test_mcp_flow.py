import subprocess
import time
import asyncio
import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

@pytest.mark.asyncio
async def test_mcp_integration():
    # Start Worker server
    worker_proc = subprocess.Popen(["python", "worker_server.py"])
    time.sleep(3)  # give server time to boot

    try:
        client = MCPClient(command=["python", "worker_server.py"])
        await client.initialize()

        tools = await client.list_tools()
        assert any(t["name"] == "summarize_youtube" for t in tools)

        result = await client.call_tool(
            "summarize_youtube",
            arguments={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        )

        assert "completed" in str(result).lower()
        await client.close()
    finally:
        worker_proc.terminate()