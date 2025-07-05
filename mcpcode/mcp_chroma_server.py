import uvicorn
# Import FastMCP from its *actual* location in the provided server.py content
from mcp.server.fastmcp.server import FastMCP

from mcp.shared.exceptions import McpError, ErrorData
from pydantic import BaseModel, Field

# Define error codes as constants
INTERNAL_ERROR_CODE = -32603
INVALID_PARAMS_CODE = -32602

import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Optional, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- ChromaDB Client Setup ---
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
CHROMA_CLIENT_TYPE = os.getenv("CHROMA_CLIENT_TYPE", "persistent")
CHROMA_DATA_DIR = os.getenv("CHROMA_DATA_DIR", "./chroma_data")

print(f"ChromaDB configuration: Type={CHROMA_CLIENT_TYPE}, Host={CHROMA_HOST}, Port={CHROMA_PORT}, DataDir={CHROMA_DATA_DIR}")

chroma_client = None
try:
    if CHROMA_CLIENT_TYPE == "http":
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        print(f"Connected to ChromaDB HTTP client at {CHROMA_HOST}:{CHROMA_PORT}")
    elif CHROMA_CLIENT_TYPE == "persistent":
        chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_DIR)
        print(f"Initialized ChromaDB persistent client at {CHROMA_DATA_DIR}")
    elif CHROMA_CLIENT_TYPE == "ephemeral":
        chroma_client = chromadb.Client()
        print("Initialized ChromaDB ephemeral client (data will not persist)")
    else:
        raise ValueError(f"Unsupported CHROMA_CLIENT_TYPE: {CHROMA_CLIENT_TYPE}")

    try:
        chroma_client.list_collections()
        print("Successfully connected to ChromaDB and listed collections.")
    except Exception as e:
        print(f"Warning: Could not list ChromaDB collections. Is the ChromaDB server running? Error: {e}")

except Exception as e:
    print(f"Error initializing ChromaDB client: {e}")
    exit(1)


# --- CORRECT FASTMCP INITIALIZATION based on *your provided server.py* ---
# FastMCP's __init__ takes host and port directly as kwargs for its internal Settings.
# The run() method then uses these settings.
mcp = FastMCP(
    sse_path="/sse",       # Uses sse_path from Settings, as per your server.py
    message_path="/messages/", # Uses message_path from Settings, as per your server.py
    host="localhost",         # Passed directly to FastMCP, used by its internal settings
    port=8001               # Passed directly to FastMCP, used by its internal settings
)

# --- Pydantic Models for Tool Outputs (Input models are flattened into tool arguments) ---
class ChromaCreateCollectionOutput(BaseModel):
    success: bool
    message: str

class ChromaAddDocumentsOutput(BaseModel):
    success: bool
    message: str

class ChromaQueryDocumentsOutput(BaseModel):
    documents: Optional[List[List[str]]] = Field(None, description="A list of lists of retrieved document texts.")
    ids: Optional[List[List[str]]] = Field(None, description="A list of lists of IDs for the retrieved documents.")
    metadatas: Optional[List[List[Dict[str, Any]]]] = Field(None, description="A list of lists of metadata dictionaries for the retrieved documents.")
    distances: Optional[List[List[float]]] = Field(None, description="A list of lists of distances for the retrieved documents.")

class ChromaDeleteCollectionOutput(BaseModel):
    success: bool
    message: str

# --- Tool Definitions (corrected to use flattened arguments) ---
@mcp.tool(name="chroma_create_collection", description="Creates a new collection in ChromaDB. Collections store documents and their embeddings.")
async def chroma_create_collection(collection_name: str = Field(description="The name of the collection to create."), metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the collection.")) -> ChromaCreateCollectionOutput:
    if not chroma_client: raise McpError(ErrorData(code=INTERNAL_ERROR_CODE, message="ChromaDB client not initialized."))
    try:
        chroma_client.create_collection(name=collection_name, metadata=metadata)
        return ChromaCreateCollectionOutput(success=True, message=f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR_CODE, message=f"Failed to create collection: {e}"))

@mcp.tool(name="chroma_add_documents", description="Adds documents to an existing ChromaDB collection.")
async def chroma_add_documents(collection_name: str = Field(description="The name of the collection to add documents to."), documents: List[str] = Field(description="A list of document texts to add."), ids: Optional[List[str]] = Field(None, description="Optional: A list of unique IDs for the documents. If not provided, ChromaDB will generate them."), metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="Optional: A list of metadata dictionaries for each document.")) -> ChromaAddDocumentsOutput:
    if not chroma_client: raise McpError(ErrorData(code=INTERNAL_ERROR_CODE, message="ChromaDB client not initialized."))
    try:
        collection = chroma_client.get_collection(name=collection_name)
        collection.add(documents=documents, ids=ids, metadatas=metadatas)
        return ChromaAddDocumentsOutput(success=True, message=f"Added {len(documents)} documents to '{collection_name}'.")
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR_CODE, message=f"Failed to add documents: {e}"))

@mcp.tool(name="chroma_query_documents", description="Queries documents in a ChromaDB collection based on similarity to a query text.")
async def chroma_query_documents(collection_name: str = Field(description="The name of the collection to query."), query_texts: List[str] = Field(description="A list of query texts to search for similarity."), n_results: int = Field(5, description="The number of results to return."), where: Optional[Dict[str, Any]] = Field(None, description="Optional: A dictionary for metadata filtering (e.g., {'source': 'article'}).")) -> ChromaQueryDocumentsOutput:
    if not chroma_client: raise McpError(ErrorData(code=INTERNAL_ERROR_CODE, message="ChromaDB client not initialized."))
    try:
        collection = chroma_client.get_collection(name=collection_name)
        results = collection.query(query_texts=query_texts, n_results=n_results, where=where, include=['documents', 'metadatas', 'distances'])
        return ChromaQueryDocumentsOutput(documents=results.get("documents", []), ids=results.get("ids", []), metadatas=results.get("metadatas", []), distances=results.get("distances", []))
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR_CODE, message=f"Failed to query documents: {e}"))

@mcp.tool(name="chroma_delete_collection", description="Deletes a collection from ChromaDB.")
async def chroma_delete_collection(collection_name: str = Field(description="The name of the collection to delete.")) -> ChromaDeleteCollectionOutput:
    if not chroma_client: raise McpError(ErrorData(code=INTERNAL_ERROR_CODE, message="ChromaDB client not initialized."))
    try:
        chroma_client.delete_collection(name=collection_name)
        return ChromaDeleteCollectionOutput(success=True, message=f"Collection '{collection_name}' deleted successfully.")
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR_CODE, message=f"Failed to delete collection: {e}"))


# --- CORRECT WAY TO RUN FastMCP (using its internal run method) ---
if __name__ == "__main__":
    print(f"Starting MCP ChromaDB Server on http://{mcp.settings.host}:{mcp.settings.port}")
    # Call FastMCP's internal run method and specify the transport
    # The 'run' method in your server.py uses self.settings.host/port directly with uvicorn.
    mcp.run(transport="sse") # This will internally call uvicorn.run(mcp.app, ...)