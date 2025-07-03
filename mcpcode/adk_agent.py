import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# --- The correct imports after clean install ---
# MCPToolset remains here
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
# StreamableHTTPConnectionParams is most reliably here based on grep
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams


# Load environment variables (for LLM API keys, etc.)
load_dotenv()

# --- ADK Agent Setup ---
async def get_chroma_agent():
    """
    Initializes and returns an ADK agent with ChromaDB tools via MCP.
    """
    mcp_server_url = "http://localhost:8001"
    print(f"Connecting to MCP server at: {mcp_server_url}")

    try:
        # Correct MCPToolset Instantiation with StreamableHTTPConnectionParams
        mcp_toolset = MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=f"{mcp_server_url}/sse",
                timeout=10,
                sse_read_timeout=10, # Include this based on grep output
            )
        )
        print("MCP tools successfully loaded from server.")
        # FIX: ADD 'await' HERE
        tools = await mcp_toolset.get_tools()
        print(f"Available MCP tools: {[tool.name for tool in tools]}")

    except Exception as e:
        print(f"Failed to connect to MCP server or load tools: {e}")
        print("Ensure 'mcp_chroma_server.py' is running on http://localhost:8001")
        return None

    llm_model = LiteLlm(model="gemini/gemini-1.5-flash") # Or your chosen Gemini model

    chroma_agent = LlmAgent(
        model=llm_model,
        name="ChromaDB_Knowledge_Agent",
        description="""
        You are a highly capable AI assistant that can interact with a ChromaDB vector database for knowledge management and retrieval.
        You can perform the following actions using your tools:
        - Create new collections: `chroma_create_collection`
        - Add documents to existing collections: `chroma_add_documents`
        - Query documents based on similarity: `chroma_query_documents`
        - Delete collections: `chroma_delete_collection`

        When asked to remember or store information, use `chroma_add_documents` to save it in a relevant collection.
        When asked a question that might require external knowledge or information you've been given, always use `chroma_query_documents` to search your knowledge base first.
        If a user asks to remember something, confirm that you have stored it.
        Provide clear and concise responses, summarizing the information retrieved from ChromaDB.
        """,
        tools=tools,
    )

    return chroma_agent

if __name__ == "__main__":
    async def test_agent_init():
        agent = await get_chroma_agent()
        if agent:
            print("\nADK Agent initialized successfully with ChromaDB tools.")
            for tool in agent.tools:
                print(f"Agent has tool: {tool.name}")
        else:
            print("\nADK Agent initialization failed.")

    asyncio.run(test_agent_init())