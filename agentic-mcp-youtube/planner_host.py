import asyncio
import json
import logging
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
from llm_client import chat_completion

# Configure logging
logging.basicConfig(level=logging.INFO, format="[PLANNER] %(message)s")

async def decide_with_llm(tools, user_goal: str):
    """
    LLM decides which tool to call + with what arguments.
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

    logging.info("Sending tool list + user goal to LLM for decision...")

    raw_text = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    logging.info(f"LLM raw output:\n{raw_text}")

    decision = json.loads(raw_text)
    logging.info(f"LLM decided → tool='{decision['tool_name']}', args={decision['arguments']}")
    return decision["tool_name"], decision["arguments"]

async def main():
    # Connect to Worker Agent
    client = MCPClient(command=["python", "worker_server.py"])
    await client.initialize()

    tools = await client.list_tools()
    logging.info(f"Discovered tools from WorkerAgent → {tools}")

    # Example user goal
    user_goal = "Summarize this YouTube video: https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # Let LLM decide
    tool_name, args = await decide_with_llm(tools, user_goal)

    # Execute via MCP
    logging.info(f"Calling MCP tool '{tool_name}' with {args}")
    result = await client.call_tool(tool_name, arguments=args)

    logging.info(f"Received Worker result → {result}")

    await client.close()

if __name__ == "__main__":
    logging.info("Starting Planner Agent...")
    asyncio.run(main())