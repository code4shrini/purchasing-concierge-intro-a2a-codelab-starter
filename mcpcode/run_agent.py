import asyncio
from adk_agent import get_chroma_agent
from google.adk.runners import Runner
from contextlib import AsyncExitStack

async def main():
    agent = await get_chroma_agent()
    if not agent:
        print("Exiting as ADK agent could not be initialized.")
        return

    # Using AsyncExitStack to properly manage the agent's lifecycle
    async with AsyncExitStack() as exit_stack:
        await exit_stack.enter_async_context(agent) # Enter the agent's context

        runner = Runner(agent)
        print("ChromaDB ADK Agent is ready. Type 'exit' to quit.")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                break

            response = await runner.run(user_input)
            print(f"Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())