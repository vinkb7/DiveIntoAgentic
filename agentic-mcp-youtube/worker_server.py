import logging
from mcp.server.stdio import stdio_server
from mcp.types import ToolCall, ToolResult
from youtube_transcript_api import YouTubeTranscriptApi
from llm_client import chat_completion

# Configure logging
logging.basicConfig(level=logging.INFO, format="[WORKER] %(message)s")

server = Server()

def fetch_transcript(video_url: str):
    """Extract transcript from YouTube video using its ID."""
    video_id = video_url.split("v=")[-1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    text = " ".join([entry["text"] for entry in transcript])
    return text

def summarize_text(text: str):
    """Summarize transcript using the configured LLM."""
    return chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful video summarizer."},
            {"role": "user", "content": f"Summarize this video transcript:\n{text}"}
        ],
        temperature=0.3,
        max_tokens=500
    )

@server.tool("summarize_youtube", input_schema={
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "YouTube video URL"}
    },
    "required": ["url"]
})
async def summarize_youtube(call: ToolCall) -> ToolResult:
    url = call.arguments["url"]
    logging.info(f"Received tool call → summarize_youtube(url={url})")

    try:
        transcript = fetch_transcript(url)
        summary = summarize_text(transcript)
        result = {"status": "completed", "url": url, "summary": summary}
    except Exception as e:
        result = {"status": "error", "message": str(e)}

    logging.info(f"Returning summary for {url}")
    return ToolResult(content=[{"type": "json", "data": result}])

if __name__ == "__main__":
    logging.info("Starting Worker MCP Server (YouTube Summarizer)...")
    stdio_server(server).run()