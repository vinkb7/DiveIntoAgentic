# planner_host.py
import asyncio
import json
import logging
import os
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from llm_client import chat_completion  # your LLM wrapper (OpenAI or LM Studio)

logging.basicConfig(level=logging.INFO, format="[PLANNER] %(message)s")
load_dotenv()


async def decide_with_llm(tools, user_goal: str):
    """
    Uses the LLM (via llm_client.chat_completion) to pick the best tool
    given the user's goal.
    """
    tool_descriptions = json.dumps(tools, indent=2)
    prompt = f"""
You are an AI planner. Available tools:

{tool_descriptions}

User goal: "{user_goal}"

Choose one tool and return valid JSON:
{{
  "tool_name": "...",
  "arguments": {{ ... }}
}}
    """
    raw_text = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    decision = json.loads(raw_text)
    return decision["tool_name"], decision["arguments"]


async def main():
    # Define how to launch Worker via stdio
    server_params = StdioServerParameters(
        command="python",
        args=["worker_server.py"],
        env=os.environ.copy()
    )

    # Start stdio client and open session
    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            logging.info("Connected to Worker MCP server.")

            # Discover tools from Worker
            tools = await session.list_tools()
            logging.info(f"Discovered tools → {tools}")

            # Example user goal
            user_goal = "Summarize this YouTube video: https://www.youtube.com/watch?v=dQw4w9WgXcQ"

            # Decide which tool + arguments using LLM
            tool_name, args = await decide_with_llm(tools, user_goal)

            # Call chosen tool
            result = await session.call_tool(tool_name, args)
            logging.info(f"Worker result → {result}")


if __name__ == "__main__":
    asyncio.run(main())