import asyncio
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
# --- The correct imports (MCPToolset from tools.mcp_tool.mcp_toolset) ---
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from mcp.client.session_group import StreamableHttpParameters
# Import SessionService ABC and ClientSession for type hinting the wrapper
from google.adk.runners import BaseSessionService # Import the ABC
from mcp.client.session import ClientSession # For type hinting get_session return
from google.adk.sessions.session import Session 
from google.adk.sessions import InMemorySessionService
# Import SseServerParameters from mcp.client.sessiongroup (as defined in your sessiongroup.py)
import litellm # Import litellm module
##litellm._turn_on_debug() # Call to turn on debugging
# Load environment variables (for LLM API keys, etc.)

load_dotenv()
# --- FORCE SET CREDENTIALS FOR DEBUGGING ---
# REMEMBER TO REMOVE THIS FOR PRODUCTION/SECURITY!
import os

# Option A: Force GOOGLE_APPLICATION_CREDENTIALS into os.environ after load_dotenv
# This is the most direct way to ensure the variable is set in Python's environment.
# Ensure this path is absolutely correct and points to your valid service account JSON key.
##os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/sudeepshrinivasan/env/sa-key.json"

# Option B: (If Option A fails) Try setting LiteLLM's specific Google API key directly
# This is usually for direct API keys, not service accounts.
# If you have a direct API key (e.g., AIzaSy...):
litellm.api_key = "AIzaSyBM-xk-mCCnLf1ZtBhgTw8XFFwZPPrf0jI"
litellm.api_key_google = "AIzaSyBM-xk-mCCnLf1ZtBhgTw8XFFwZPPrf0jI" # Sometimes specific to provider

# Let's stick with Option A first, as it's designed for service accounts.

# --- Custom SessionService Wrapper ---
# This class adapts MCPToolset's internal session management
# to fit the SessionService interface expected by google.adk.runners.Runner.
# --- Custom SessionService Wrapper ---
# This class adapts MCPToolset's internal session management
# to fit the SessionService interface expected by google.adk.runners.Runner.
# class CustomSessionService(BaseSessionService):
#     def __init__(self, mcp_toolset_instance: MCPToolset):
#         self._mcp_toolset = mcp_toolset_instance
#         self._internal_session_manager = mcp_toolset_instance._mcp_session_manager

#     # --- FIX: Access _sessions correctly as a dictionary ---
#     async def get_session(self, user_id: str, session_id: str, app_name: str) -> Session:
#         """
#         Gets or creates an underlying MCP session and returns it wrapped in an ADK Session object.
#         """
#         # Use the MCP session manager to get or create the underlying MCP session.
#         headers = {user_id: user_id, session_id: session_id}
#         mcp_session = await self._internal_session_manager.create_session(
#             headers=headers,)
        
#         # Wrap the information into the ADK `Session` object that the Runner expects.
#         return Session(
#             user_id=mcp_session.__getstate__.user_id,
#             session_id=mcp_session.session_id,
#             app_name=app_name,
#         )


#     # Implement remaining abstract methods from SessionService
#     async def create_session(self, user_id: str, session_id: str) -> Session:
#         print(f"CustomSessionService: Attempting to create/get session for user='{user_id}', session='{session_id}'")
#         return await self.get_session(user_id, session_id, app_name="DefaultAppForCreate")

#     async def delete_session(self, user_id: str, session_id: str) -> None:
#         print(f"CustomSessionService: Deleting session for user='{user_id}', session='{session_id}'")
#         pass

#     # --- FIX: Iterate _sessions dictionary keys (which are ClientSession objects) ---
#     async def list_sessions(self, user_id: str) -> List[str]:
#         """
#         Lists active session IDs for a user. For this simple runner, it returns the current session ID if active.
#         """
#         print(f"CustomSessionService: Listing sessions for user='{user_id}'")
#         # FIX HERE: Iterate over the keys of the _sessions dictionary (which are ClientSession objects).
#         if self._internal_session_manager._sessions:
#              # Filter sessions by the provided user_id
#             return [
#                 s.session_id
#                 for s in self._internal_session_manager._sessions.keys()
#                 if s.user_id == user_id
#             ]
#         return []

# --- ADK Agent Setup ---
async def get_chroma_agent() -> Tuple[LlmAgent, MCPToolset, InMemorySessionService]:
    """
    Initializes and returns an ADK agent with ChromaDB tools via MCP.
    """
    mcp_server_base_url = "http://localhost:8001" # This is the base of the server
    # FastMCP's default sse_path is /sse, so client needs to connect to that specific endpoint
    mcp_server_sse_url = f"{mcp_server_base_url}/sse"
    print(f"Connecting to MCP server at: {mcp_server_sse_url}")

    try:
        # Correct MCPToolset Instantiation with SseServerParameters
        mcp_toolset = MCPToolset(
            connection_params= SseConnectionParams (# Use SseServerParameters from mcp.client.sessiongroup
                url=mcp_server_sse_url, # Client URL MUST include the /sse endpoint
                timeout=30, # Passed as float
                sse_read_timeout=30 # Passed as float
            )
        )
        print("MCP tools successfully loaded from server.")
        tools = await mcp_toolset.get_tools() # Already correctly awaited
        print(f"Available MCP tools: {[tool.name for tool in tools]}")
        # Create the custom session service wrapper
        custom_session_service = InMemorySessionService()

    except Exception as e:
        print(f"Failed to connect to MCP server or load tools: {e}")
        print("Ensure 'mcp_chroma_server.py' is running on http://localhost:8001")
        return None

    llm_model = LiteLlm(model="gemini/gemini-1.5-flash") # Or your chosen Gemini model (e.g., "gemini/gemini-pro")

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

    return chroma_agent, mcp_toolset, custom_session_service

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