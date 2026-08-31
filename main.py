"""Minimal OpenHands SDK agent backed by the local vLLM server."""

import os

from openhands.sdk import Conversation
from openhands.tools.preset.default import get_default_agent

from config import default_profile

llm = default_profile().build("local-qwen")

agent = get_default_agent(llm=llm, cli_mode=True)

conversation = Conversation(agent=agent, workspace=os.getcwd())
try:
    conversation.send_message(
        "Create a file named hello.txt containing a 5-line greeting, then read it back."
    )
    conversation.run()
finally:
    conversation.close()
