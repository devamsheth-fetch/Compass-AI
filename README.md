# CompassAI: Value-Based Local Guide Multi-Agent System

CompassAI is a fully decentralized, multi-agent AI travel and location advisory system built on **Fetch.ai's uAgents framework** and powered by **Google Gemma-3** via the **Agent Developer Kit (ADK)**. 

The system operates as a swarm of specialized agents that gather constraints from the user, scrape real-time location and weather data, fetch the latest news/facts, and synthesize everything into a tailored itinerary. It integrates natively with the **ASI1 / DeltaV** platforms to provide an interactive, conversational user experience.

## System Architecture

The project is structured as three completely decoupled `uAgent` nodes running on different ports that communicate asynchronously via predefined Pydantic models:

1. **The Guide (`agents/guide.py`) [Port: 8000]**
   - The primary conversational orchestrator.
   - Interacts directly with the user over DeltaV to gather exactly 5 constraints (Activity, Location, Time, Budget, Specific Needs).
   - Once all constraints are gathered, it hands off the raw JSON over the network to the Travel Advisor.
   
2. **The Travel Advisor (`agents/travel_advisor_agent.py`) [Port: 8001]**
   - Receives constraints and uses the **Yelp API** to find 5 matching locations with user reviews.
   - Uses the **OpenWeatherMap API** to get real-time weather conditions for the area.
   - Forwards an aggregated context payload to the Trip Planner.

3. **The Trip Planner (`agents/planner_agent.py`) [Port: 8002]**
   - Receives the raw location data and weather conditions.
   - Uses the **Tavily API** to fetch the latest local news and interesting facts about the chosen spots.
   - Employs its LLM to calculate estimated costs and generates a singular, perfectly optimized itinerary.
   - Sends the finalized itinerary directly back to The Guide for delivery to the user.

*(Note: The Planner can also asyncronously ask the user clarifying questions mid-workflow by routing questions back through The Guide!)*

## Prerequisites

- Python 3.10+
- `pip` or Poetry for dependency management

### Install Dependencies

```bash
pip install uagents google-adk requests python-dotenv uagents-core
```

### Environment Variables

You need to provide several API keys and Fetch.ai Mailbox keys to run the network. Create a `.env` file in the root directory and configure it as follows:

```env
# AI & Search APIs
GEMINI_API_KEY="your_gemini_api_key_here"
OPENWEATHER_API_KEY="your_openweather_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"
YELP_API_KEY="your_yelp_api_key_here"

# Agentverse Mailbox Keys (for DeltaV deployment)
GUIDE_MAILBOX_KEY="your_guide_mailbox_key_here"
ADVISOR_MAILBOX_KEY="your_advisor_mailbox_key_here"
PLANNER_MAILBOX_KEY="your_planner_mailbox_key_here"

# Cryptographic Agent Addresses (automatically derived from SEEDS in shared_config.py)
GUIDE_ADDRESS="agent1..."
ADVISOR_ADDRESS="agent1..."
PLANNER_ADDRESS="agent1..."
```

*Note: Ensure your `shared_config.py` correctly mirrors the seeds used to generate the Agentverse Mailbox Keys.*

## Running the Multi-Agent Swarm

Since the agents operate asynchronously over the Fetch.ai network, they each need to be run in their own terminal instance. 

Open three separate terminals and run the following commands from the root directory:

**Terminal 1 (The Planner):**
```bash
python agents/planner_agent.py
```

**Terminal 2 (The Travel Advisor):**
```bash
python agents/travel_advisor_agent.py
```

**Terminal 3 (The Guide):**
```bash
python agents/guide.py
```

Once all three agents are running and connected to their respective Agentverse Mailboxes, you can engage with the system over ASI1 / DeltaV by searching for your registered Guide Agent.

## File Structure

```
CompassAI/
├── agents/
│   ├── guide.py                  # The user-facing conversational agent
│   ├── travel_advisor_agent.py   # The location and weather data aggregator
│   └── planner_agent.py          # The final itinerary generator and fact-checker
├── tools/
│   └── core_tools.py             # Python functions wrapping Yelp, OpenWeather, and Tavily APIs
├── shared_config.py              # Cryptographic Seeds, Network Addresses, and Pydantic Message Models
├── .env                          # API & Mailbox Keys (Ignored by Git)
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```
