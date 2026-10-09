"""Week 5 - Day 1: AutoGen HR Assistant Agent.

Asks an AssistantAgent for a 5-line job description for a Junior Python Developer.

Run:
    pip install -U "autogen-agentchat" "autogen-ext[openai]"
    export OPENAI_API_KEY="your-key"      # PowerShell: $env:OPENAI_API_KEY="your-key"
    python hr_agent.py
"""
import asyncio
from dotenv import load_dotenv
load_dotenv()
import asyncio
import os
import sys

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# "5 lines" is ambiguous to a model, so the task spells out exactly what a line is.
TASK = (
    "Write a 5-line job description for a Junior Python Developer.\n"
    "Format: put the job title on its own line, then exactly 5 numbered lines "
    "(1. to 5.), each a single sentence.\n"
    "Do not add an introduction, a closing remark, blank lines, or markdown formatting."
)


async def main() -> None:
    # The key comes from the environment, never from the source code.
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set. Export it before running this script.")

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

    agent = AssistantAgent(
        name="hr_assistant",
        model_client=model_client,
        system_message="You are an HR assistant.",
    )

    try:
        # Console prints each message from the stream as it arrives.
        await Console(agent.run_stream(task=TASK))
    finally:
        # Always close the client, even on error, or the script hangs on exit.
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())