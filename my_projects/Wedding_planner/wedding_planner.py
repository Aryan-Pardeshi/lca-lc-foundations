# multi agent wedding planner
import os
import asyncio
from datetime import datetime
from typing import Dict, Any


from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

# Suppress annoying schema warnings from MCP / Langchain
import logging
import warnings
class SchemaWarningFilter(logging.Filter):
    def filter(self, record):
        return "not supported in schema, ignoring" not in record.getMessage()
logging.getLogger().addFilter(SchemaWarningFilter())
warnings.filterwarnings("ignore", message=".*not supported in schema, ignoring.*")


#state for the agent to store information
class WeddingState(AgentState):
    origin: str
    destination: str
    date: str
    guest_count: str
    genre: str


from tavily import TavilyClient
from langchain_community.utilities import SQLDatabase

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
tavily_client = TavilyClient()

# MCP Clients
flight_client = MultiServerMCPClient(
    {"travel_server": {"transport": "streamable_http", "url": "https://mcp.kiwi.com"}}
)

venue_client = MultiServerMCPClient(
    {
        "firecrawl-mcp": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "firecrawl-mcp"],
            "env": {"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")},
        }
    }
)

# Playlist Database
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Chinook.db")
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH.replace(os.sep, '/')}")


@tool
async def get_playlist_by_genre(genre: str) -> str:
    """Get a list of songs matching a specific music genre from the database."""
    try:
        query = f"SELECT t.Name as Track, a.Name as Artist FROM Track t JOIN Album al ON t.AlbumId = al.AlbumId JOIN Artist a ON al.ArtistId = a.ArtistId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name LIKE '%{genre}%' LIMIT 15;"
        return db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"


async def build_agent():
    # Flight subagent
    flight_tool = await flight_client.get_tools()

    flight_agent = create_agent(
        model=model,
        tools=flight_tool,
        system_prompt="You are a flight finder for weddings. Find flights and return only the required flight details in JSON format.",
    )

    @tool
    async def flights_searcher(query: str) -> str:
        """Search for flights based on the user's request. Returns flight details in JSON."""
        response = await flight_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=query
                        + " — return only required flight details in JSON format."
                    )
                ]
            }
        )
        msg = response["messages"][-1]
        return msg.content[0]["text"] if isinstance(msg.content, list) else msg.content

    # Venue subagent
    venue_tool = await venue_client.get_tools()

    @tool
    async def search_venues(query: str) -> list:
        """Search the web for wedding venue URLs using the query. Returns a list of relevant URLs."""
        results = tavily_client.search(query + " wedding venues")
        return [r["url"] for r in results.get("results", [])]

    venue_agent = create_agent(
        model=model,
        tools=[search_venues] + venue_tool,
        system_prompt=(
            "You are a wedding venue researcher. Follow these steps strictly:\n"
            "1. Use 'search_venues' to find a few wedding venue URLs.\n"
            "2. Pick ONE relevant URL and pass it to 'firecrawl_scrape'.\n"
            "CRITICAL FIRECRAWL INSTRUCTION: You MUST ONLY pass the `url` parameter to 'firecrawl_scrape'. "
            "DO NOT PASS `jsonOptions`, DO NOT pass `schema`. If you pass anything other than `url`, the system will crash!\n"
            "3. Extract name, location, capacity, and price from the scraped content and return it."
        ),
    )

    @tool
    async def venues_searcher(query: str) -> str:
        """Search for wedding venues based on the user's request. Returns venue details in JSON."""
        try:
            response = await venue_agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=query
                        )
                    ]
                }
            )
            msg = response["messages"][-1]
            return msg.content[0]["text"] if isinstance(msg.content, list) else msg.content
        except Exception as e:
            return f"Error talking to Venue tool. Tell the user it failed temporarily. Details: {str(e)}"

    # Playlist Database is loaded globally as query_playlist_db

    # Tavily
    @tool
    async def web_search(query: str) -> Dict[str, Any]:
        """Search the web for information."""
        return tavily_client.search(query)

    # Main agent
    now = datetime.now().strftime("%A, %d %B %Y %I:%M %p")

    main_agent = create_agent(
        model=model,
        tools=[flights_searcher, venues_searcher, get_playlist_by_genre, web_search],
        system_prompt=(
            f"You are an expert, highly organized Wedding Planner AI Assistant. Today is {now}.\n\n"
            "--- YOUR PERSONA ---\n"
            "You are friendly, empathetic, yet highly efficient. Planning a wedding is stressful "
            "for the user, so you handle the logistics with confidence and clarity. "
            "You never guess information; you always verify requirements first.\n\n"
            "--- REQUIRED INFORMATION (STATE) ---\n"
            "To plan effectively, you MUST collect the following FIVE pieces of information:\n"
            "1. Origin (for flights)\n"
            "2. Destination (for flights and venues)\n"
            "3. Date (for flight bookings)\n"
            "4. Guest Count (for venues)\n"
            "5. Music Genre (for Spotify playlists)\n\n"
            "--- WORKFLOW ---\n"
            "Step 1: Check your State. If ANY of the 5 required pieces of information are missing, "
            "you MUST politely ask the user to provide them. Do NOT invoke any tools (except maybe web_search for general chat) "
            "until you have all 5 pieces of information.\n"
            "Step 2: Once you have all 5 pieces of data, summarize the plan for the user, and begin using your tools to fetch real suggestions.\n"
            "Step 3: Present your findings in a beautifully structured format, using bullet points or markdown tables. "
            "Ensure you offer 2-3 flight options, 2-3 venue options, and a curated playlist.\n"
            "Step 4: Ask the user for feedback or if they need adjustments (e.g., cheaper flights, larger venues).\n\n"
            "--- TOOL USAGE & CONSTRAINTS ---\n"
            "- flights_searcher: Use when you know Origin, Destination, and Date.\n"
            "- venues_searcher: Use when you know Destination and Guest Count.\n"
            "- get_playlist_by_genre: Use when you know the Music Genre.\n"
            "- web_search: Use for general wedding queries (e.g., 'What is a typical wedding dress budget?'). "
            "You MUST ALSO use `web_search` as a BACKUP fallback. If `flights_searcher`, `venues_searcher`, or `get_playlist_by_genre` return an error, fail, or can't find anything, immediately retry your search using `web_search` to ensure the user still gets a good answer!\n\n"
            "Remember: Your ultimate goal is to generate a comprehensive, actionable Wedding Plan. "
            "Take it one step at a time, and ensure you have all required data before acting!"
        ),
        state_schema=WeddingState,
        checkpointer=InMemorySaver(),
    )
    return main_agent


async def main():
    main_agent = await build_agent()
    config = {"configurable": {"thread_id": "wedding-planner-1"}}

    print("\n" + "═" * 70)
    print(" 💖 Wedding Planner Agent ready! (Type 'exit' to quit) ")
    print("═" * 70 + "\n")

    try:
        while True:
            # Solid line before user turn
            print("─" * 70)
            user_input = input("\033[94m You: \033[0m").strip()

            if user_input.lower() in ("exit", "quit"):
                print("─" * 70)
                print("\n\033[95m Goodbye! Happy planning!\033[0m\n")
                break
            if not user_input:
                continue

            # Yellow loading message because search could take a few seconds
            print(
                "\n\033[33m Planning (this may take a moment while I search)...\033[0m"
            )
            print("─" * 70)

            response = await main_agent.ainvoke(
                {"messages": [HumanMessage(content=user_input)]}, config
            )

            # Green text for agent
            print("\n\033[92m Agent:\033[0m")
            msg_content = response['messages'][-1].content
            agent_text = msg_content[0]["text"] if isinstance(msg_content, list) else msg_content
            print(f"{agent_text}\n")

    except KeyboardInterrupt:
        print("\n" + "─" * 70)
        print("\n\033[95m Goodbye! Happy planning!\033[0m\n")


if __name__ == "__main__":
    print("Loading Wedding Planner Agent....")
    asyncio.run(main())

# example prompt:
# "Hi! I am located in Mumbai and I'm planning an high-budget wedding. We will be flying our families out, so our origin is Mumbai and our destination is Udaipur. We are expecting a large gathering with a guest count of 500. For the Sangeet and reception, we definitely need a high-energy Bollywood and Punjabi Pop music genre."
