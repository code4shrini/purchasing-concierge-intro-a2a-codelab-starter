import asyncio
from adk_agent import get_chroma_agent
from google.genai import types
from google.adk.runners import Runner, RunConfig
import uuid # For generating unique session_id
from google.adk.sessions.session import Session
from google.adk.sessions import InMemorySessionService
from typing import Any, Iterator # Ensure Iterator is imported for the helper function
 # Import Part for message construction

# Helper function to run a synchronous generator in a separate thread and yield its results asynchronously
async def _async_yield_from_sync_generator(sync_gen: Iterator[Any]) -> Any:
    """
    Wraps a synchronous generator to yield its items asynchronously,
    running each 'next' call in a separate thread.
    """
    while True:
        try:
            # We use a helper function to wrap next(sync_gen) so that
            # StopIteration is caught and a specific signal (None) is returned
            # instead of raising StopIteration directly in the async context.
            def _get_next_item():
                try:
                    return next(sync_gen)
                except StopIteration:
                    return None # Signal end of iteration

            item = await asyncio.to_thread(_get_next_item)
            if item is None: # Check for our sentinel value
                break
            yield item
        except Exception as e:
            print(f"Error in sync generator: {e}")
            raise

async def main():
    agent, mcp_tool_set, custom_session_service = await get_chroma_agent()
    if not agent or not mcp_tool_set or not custom_session_service:
        print("Exiting as ADK agent could not be initialized.")
        return

    runner = Runner(
        app_name="ChromaDB_Agent_App", # A descriptive name for your application
        agent=agent,
        session_service=custom_session_service # Pass the custom_session_service here
    )
    print("ChromaDB ADK Agent is ready. Type 'exit' or 'quit' when prompted by the agent to end the session.")

    user_id = "default_user" # Define a user ID for the session

    inMemSession= await custom_session_service.create_session(
        app_name="ChromaDB_Agent_App", # The same app name used in Runner
        user_id=user_id, # Use the defined user ID
        state=None, # No initial state needed for this example
        session_id=str(uuid.uuid4()) # Generate a unique session ID
    )
    session_id =inMemSession.id # Generate a unique session ID for this run

    while True:
        user_input = input("\nYou: ") # Prompt the user for input
        if user_input.lower() == 'exit' or user_input.lower() == 'quit':
            break # Exit the loop if user types 'exit' or 'quit'

        user_message = types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
        run_config = RunConfig(response_modalities=["TEXT"])
        print(f"[DEBUG] RunConfig: {run_config}")  # Log the configuration
        try:
            sync_generator_output = runner.run(
                user_id=user_id,
                session_id=session_id,
                new_message=user_message,  # Correct: pass as dict
                run_config=run_config,  # Pass the RunConfig object
            )


            # Use the helper to consume the synchronous generator asynchronously
            full_response_parts = []
            async for response_chunk in _async_yield_from_sync_generator(sync_generator_output):
                full_response_parts.append(str(response_chunk)) # Convert chunk to string for joining
            final_response = "".join(full_response_parts)
            print(f"Agent: {final_response}")

        except Exception as e:
            # Catch any exceptions that occur during the interaction process.
            print(f"Agent: An error occurred during interaction: {e}")
            # This is where the pydantic ValidationError will be caught.
            break # Exit the loop if an unhandled error occurs during interaction

    print("Agent session ended.")

if __name__ == "__main__":
    asyncio.run(main())