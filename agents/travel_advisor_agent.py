import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from uagents import Agent, Context, Protocol
from datetime import datetime, timezone
from uuid import uuid4

from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)
from shared_config import ADVISOR_SEED, PLANNER_ADDRESS, ConstraintsPayload, LocationOptionsPayload, UpdatedConstraintsPayload, UserPromptPayload
from tools.core_tools import get_location_data, get_weather_data

load_dotenv()
session_service = InMemorySessionService()

travel_advisor_agent = LlmAgent(
    name="travel_advisor",
    model="gemma-3-27b-it",
    description="The Travel Advisor sub-agent. Gathers location options and live weather.",
    instruction="""
    You are the "Travel Advisor" sub-agent.
    
    [DIRECT CHAT MODE]
    If a user is chatting with you directly, answer their questions naturally in plain text. Your domain is location options, user reviews, and weather data. If a user asks you about a specific place or area (e.g. "Do you know anything about communication hills?"), proactively use your `get_location_data` and `get_weather_data` tools to look up the area and tell them what you found (cool spots, weather, reviews)! DO NOT state your limitations, DO NOT apologize for not knowing history, and DO NOT mention what you do not know. Simply and enthusiastically provide the location and weather data you gathered. DO NOT act like a trip planner or try to build full itineraries. DO NOT mention "Context Packages".
    
    [SYSTEM WORKFLOW MODE]
    However, when you are explicitly given a specific set of JSON routing constraints to process a trip:
    Your specific purpose here is ONLY providing location options, user reviews, and weather data. DO NOT provide general facts, trivia, or history about locations.
    1. ALWAYS call the `get_location_data` tool to find 5 relevant places to visit, eat, or do activities based on the constraints.
    2. ALWAYS call the `get_weather_data` tool to find the current conditions for that area. VERY IMPORTANT: Pass ONLY the broad city or region name (e.g., "San Francisco", "Austin") to the weather tool. Do NOT pass specific park or business names to the weather tool, as it will fail to find them.
    
    In this System Workflow Mode ONLY, combine this data into a clear 'Context Package' text summary, detailing the spots, their distance/ratings, and the weather, and provide this summary as your final output. DO NOT include any extra general facts or trivia.
    """,
    tools=[get_location_data, get_weather_data]
)

advisor_uagent = Agent(
    name="travel_advisor_uagent",
    port=8001,
    seed=ADVISOR_SEED,
    mailbox=os.getenv("ADVISOR_MAILBOX_KEY")
)



@advisor_uagent.on_message(model=ConstraintsPayload)
async def handle_constraints(ctx: Context, sender: str, msg: ConstraintsPayload):
    ctx.logger.info(f"Received Constraints from {sender}")
    constraints = json.loads(msg.constraints_json)
    
    # Run the local ADK Runner for the Travel Advisor
    runner = Runner(
        app_name="advisor_app",
        agent=travel_advisor_agent,
        session_service=session_service,
        auto_create_session=True
    )
    
    prompt = f"Find 5 places to visit, eat, or do activities based on these constraints: {json.dumps(constraints)}"
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    
    ctx.logger.info("-> Waking up The Travel Advisor Agent...")
    advisor_response = ""
    async for event in runner.run_async(
        user_id=msg.user_sender,
        session_id=msg.user_sender,
        new_message=message
    ):
        if hasattr(event, 'content') and event.content and isinstance(event.content, types.Content):
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    advisor_response += part.text
        elif hasattr(event, 'text') and event.text:
            advisor_response += event.text
            
    ctx.logger.info("<- Travel Advisor Agent Finished. Forwarding to Planner...")
    
    # Forward the results to the Planner
    payload = LocationOptionsPayload(
        user_sender=msg.user_sender,
        constraints_json=msg.constraints_json,
        advisor_text=advisor_response
    )
    await ctx.send(PLANNER_ADDRESS, payload)
    ctx.logger.info(f"Sent LocationOptions to Planner at {PLANNER_ADDRESS}")

@advisor_uagent.on_message(model=UpdatedConstraintsPayload)
async def handle_clarification(ctx: Context, sender: str, msg: UpdatedConstraintsPayload):
    ctx.logger.info(f"Received Updated Constraints (Clarification) from {sender}")
    constraints = json.loads(msg.constraints_json)
    
    runner = Runner(
        app_name="advisor_app",
        agent=travel_advisor_agent,
        session_service=session_service,
        auto_create_session=True
    )
    
    prompt = f"The user has provided a clarification or added a new requirement: '{msg.user_clarification}'.\nTheir original constraints were: {json.dumps(constraints)}.\nPlease use your location tool to find 1 or 2 new places that specifically address this new requirement (for example, finding a restaurant if they just added they are hungry). DO NOT fetch weather again."
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    
    ctx.logger.info("-> Waking up The Travel Advisor Agent for Clarification...")
    advisor_response = ""
    async for event in runner.run_async(
        user_id=msg.user_sender,
        session_id=msg.user_sender + "_clarification", # Unique session so it doesn't cross wires
        new_message=message
    ):
        if hasattr(event, 'content') and event.content and isinstance(event.content, types.Content):
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    advisor_response += part.text
        elif hasattr(event, 'text') and event.text:
            advisor_response += event.text
            
    ctx.logger.info("<- Clarification Finished. Combining with previous options and forwarding to Planner...")
    
    # Forward BOTH the old locations and the new locations to the Planner
    combined_text = f"--- Original Options ---\n{msg.previous_advisor_text}\n\n--- New Options Based On User Clarification ---\n{advisor_response}"
    
    payload = LocationOptionsPayload(
        user_sender=msg.user_sender,
        constraints_json=msg.constraints_json,
        advisor_text=combined_text
    )
    await ctx.send(PLANNER_ADDRESS, payload)
    ctx.logger.info(f"Sent augmented LocationOptions to Planner at {PLANNER_ADDRESS}")

@advisor_uagent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"Travel Advisor uAgent starting. Address: {advisor_uagent.address}")
    ctx.logger.info(f"Local HTTP Endpoint enabled. Bypassing Mailboxes.")

@advisor_uagent.on_message(model=UserPromptPayload)
async def handle_user_prompt(ctx: Context, sender: str, msg: UserPromptPayload):
    ctx.logger.info(f"Received raw UserPromptPayload from {sender}: {msg.text}")
    
    runner = Runner(
        app_name="advisor_app",
        agent=travel_advisor_agent,
        session_service=session_service,
        auto_create_session=True
    )
    
    prompt = types.Content(role="user", parts=[types.Part.from_text(text=msg.text)])
    
    ctx.logger.info("-> Waking up The Travel Advisor Agent for a Direct Prompt...")
    advisor_response = ""
    async for event in runner.run_async(
        user_id=sender,
        session_id=sender,
        new_message=prompt
    ):
        if hasattr(event, 'content') and event.content and isinstance(event.content, types.Content):
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    advisor_response += part.text
        elif hasattr(event, 'text') and event.text:
            advisor_response += event.text
            
    ctx.logger.info("<- Travel Advisor Agent Finished Direct Prompt.")
    
    # Send the response back as another UserPromptPayload so the sender can read the string natively
    response_payload = UserPromptPayload(text=advisor_response)
    await ctx.send(sender, response_payload)
    ctx.logger.info(f"Sent response back to {sender}")

# --- Direct User Chat Protocol ---
advisor_protocol = Protocol(spec=chat_protocol_spec)

def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent())

    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=content,
    )

@advisor_protocol.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"Received Message from {sender}")
    
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id
        )
    )
    
    for item in msg.content:
        if isinstance(item, StartSessionContent):
            pass
        elif isinstance(item, TextContent):
            text = item.text.strip()
            import re
            text = re.sub(r'^@\S+\s+', '', text).strip()
            if not text:
                continue
                
            runner = Runner(
                app_name="advisor_app",
                agent=travel_advisor_agent,
                session_service=session_service,
                auto_create_session=True
            )
            message = types.Content(role="user", parts=[types.Part.from_text(text=text)])
            
            ctx.logger.info("-> Waking up The Travel Advisor Agent for Direct Chat...")
            full_response = ""
            async for event in runner.run_async(
                user_id=sender,
                session_id=sender,
                new_message=message
            ):
                if hasattr(event, 'content') and event.content and isinstance(event.content, types.Content):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            full_response += part.text
                elif hasattr(event, 'text') and event.text:
                    full_response += event.text
            
            await ctx.send(sender, create_text_chat(full_response))

        elif isinstance(item, EndSessionContent):
            await ctx.send(sender, create_text_chat("Goodbye!", end_session=True))

@advisor_protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received ChatAck from {sender} for message {msg.acknowledged_msg_id}")

advisor_uagent.include(advisor_protocol, publish_manifest=True)

if __name__ == "__main__":
    advisor_uagent.run()
