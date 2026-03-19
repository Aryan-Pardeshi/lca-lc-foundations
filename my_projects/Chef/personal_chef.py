from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

from langchain.messages import HumanMessage

from langchain.tools import tool

from langgraph.checkpoint.memory import InMemorySaver  

from tavily import TavilyClient
from typing import Dict, Any

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for the query and return the results."""
    return tavily_client.search(query)

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt="You are a chef and you help people find recipes based on their requests.",
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}} #1st chat

import base64
import os

def main():
    print("How may I help you with your cooking today? (type 'exit' to quit)")
    try:
        while True:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            # Ask for optional image path
            image_path = input("Enter image path (or press Enter to skip): ").strip()

            img_b64 = None
            if image_path:
                if not os.path.exists(image_path):
                    print("Image not found; continuing without image.")
                else:
                    with open(image_path, "rb") as f:
                        img_bytes = f.read()
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            # Build human message (include image if provided)
            if img_b64:
                human_content = [
                    {"type": "text", "text": question},
                    {"type": "image", "base64": img_b64, "mime_type": "image/png"},
                ]
            else:
                human_content = [{"type": "text", "text": question}]

            query = HumanMessage(content=human_content)

            response = agent.invoke({"messages": [query]}, config=config)

            # Print assistant reply
            try:
                print(response["messages"][-1].content)
            except Exception:
                print(response)

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()