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
from shared_config import PLANNER_SEED, GUIDE_ADDRESS, LocationOptionsPayload, FinalItineraryPayload, PlannerQuestionPayload, UserPromptPayload
from tools.core_tools import search_local_facts

load_dotenv()
session_service = InMemorySessionService()

def ask_clarifying_question(question: str):
    """
    Call this tool if the itinerary feels incomplete and you want to ask the user for clarifying details (e.g., "Would you also like me to find a place to eat after your hike?").
    """
    return f"PLANNER_QUESTION_TRIGGERED: {question}"

planner_agent = LlmAgent(
    name="trip_planner",
    model="gemma-3-27b-it",
    description="The Trip Planner sub-agent. Finalizes full itinerary with costs.",
    instruction="""
    You are the "Trip Planner" agent.
    
    [DIRECT CHAT MODE]
    If a user is chatting with you directly, answer their questions normally in plain text. Your specific purpose is ONLY gathering facts and the latest news about locations they ask about. ALWAYS use the `search_local_facts` tool to look for the latest news and facts around the location or region the user mentions. DO NOT ask the user questions about their trip (like budget, who they are traveling with, lengths of stay). DO NOT explain to the user how you "plan", how your tools work, or what your role is in the system. Just provide the interesting facts and news they asked for naturally. DO NOT try to generate a full itinerary or estimate costs if they are just chatting with you.
    
    [SYSTEM WORKFLOW MODE]
    However, when you receive system constraints and location options from the Travel Advisor to coordinate a trip:
    If the trip seems to be missing a logical component based on the location options (for example, the user wants to go hiking, but didn't mention food, and you think they might be hungry afterward), you MUST call the `ask_clarifying_question` tool to ask them.
    If you call that tool, do not output anything else.
    
    Otherwise:
    Step 1: ALWAYS use the `search_local_facts` tool (Tavily API) to fetch relevant facts and the LATEST NEWS about the provided locations or regions.
    Step 2: Design a singular, finalized full itinerary incorporating the best places from the provided options, integrating any relevant latest news and facts. Do NOT provide multiple options or separate lists; give them ONE perfectly planned route.
    Step 3: Include a final estimated cost based on the budget and places chosen.
    """,
    tools=[search_local_facts, ask_clarifying_question]
)

planner_uagent = Agent(
    name="trip_planner_uagent",
    port=8002,
    seed=PLANNER_SEED,
    mailbox=os.getenv("PLANNER_MAILBOX_KEY")
)

@planner_uagent.on_message(model=LocationOptionsPayload)
async def handle_locations(ctx: Context, sender: str, msg: LocationOptionsPayload):
    ctx.logger.info(f"Received Location Options from {sender}")
    constraints = json.loads(msg.constraints_json)
    
    runner = Runner(
        app_name="planner_app",
        agent=planner_agent,
        session_service=session_service,
        auto_create_session=True
    )
    
    prompt = f"Here are the constraints: {json.dumps(constraints)}\n\nHere are the location options gathered by the Travel Advisor:\n{msg.advisor_text}\n\nPlease fetch any relevant facts via Tavily, and generate the final itinerary and estimated cost."
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    
    ctx.logger.info("-> Waking up The Planner Agent...")
    planner_response = ""
    async for event in runner.run_async(
        user_id=msg.user_sender,
        session_id=msg.user_sender,
        new_message=message
    ):
        if hasattr(event, 'content') and event.content and isinstance(event.content, types.Content):
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    planner_response += part.text
        elif hasattr(event, 'text') and event.text:
            planner_response += event.text
            
    ctx.logger.info("<- Planner Agent Finished.")
    
    if "PLANNER_QUESTION_TRIGGERED:" in planner_response:
        question = planner_response.split("PLANNER_QUESTION_TRIGGERED:")[1].strip()
        ctx.logger.info(f"Planner has a question for the user: {question}")
        payload = PlannerQuestionPayload(
            user_sender=msg.user_sender,
            constraints_json=msg.constraints_json,
            advisor_text=msg.advisor_text,
            question_text=question
        )
        await ctx.send(GUIDE_ADDRESS, payload)
        ctx.logger.info(f"Sent PlannerQuestion back to {GUIDE_ADDRESS}")
    else:
        ctx.logger.info("Sending Final Itinerary back to Guide...")
        payload = FinalItineraryPayload(
            user_sender=msg.user_sender,
            itinerary_text=planner_response
        )
        await ctx.send(GUIDE_ADDRESS, payload)
        ctx.logger.info(f"Sent FinalItinerary back to {GUIDE_ADDRESS}")

@planner_uagent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"Planner uAgent starting. Address: {planner_uagent.address}")
    ctx.logger.info(f"Local HTTP Endpoint enabled. Bypassing Mailboxes.")

@planner_uagent.on_message(model=UserPromptPayload)
async def handle_user_prompt(ctx: Context, sender: str, msg: UserPromptPayload):
    ctx.logger.info(f"Received raw UserPromptPayload from {sender}: {msg.text}")
    
    runner = Runner(
        app_name="planner_app",
        agent=planner_agent,
        session_service=session_service,
        auto_create_session=True
    )
    
    prompt = types.Content(role="user", parts=[types.Part.from_text(text=msg.text)])
    
    ctx.logger.info("-> Waking up The Planner Agent for a Direct Prompt...")
    planner_response = ""
    async for event in runner.run_async(
        user_id=sender,
        session_id=sender,
        new_message=prompt
    ):
        if hasattr(event, 'content') and event.content and isinstance(event.content, types.Content):
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    planner_response += part.text
        elif hasattr(event, 'text') and event.text:
            planner_response += event.text
            
    ctx.logger.info("<- Planner Agent Finished Direct Prompt.")
    
    # Send the response back as another UserPromptPayload so the sender can read the string natively
    response_payload = UserPromptPayload(text=planner_response)
    await ctx.send(sender, response_payload)
    ctx.logger.info(f"Sent response back to {sender}")

# --- Direct User Chat Protocol ---
planner_protocol = Protocol(spec=chat_protocol_spec)

def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent())

    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=content,
    )

@planner_protocol.on_message(ChatMessage)
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
            # await ctx.send(sender, create_text_chat("Hello! I am the Trip Planner sub-agent. Give me some raw facts and I'll generate a beautiful itinerary for you!"))
            pass
        elif isinstance(item, TextContent):
            text = item.text.strip()
            import re
            text = re.sub(r'^@\S+\s+', '', text).strip()
            if not text:
                continue
                
            runner = Runner(
                app_name="planner_app",
                agent=planner_agent,
                session_service=session_service,
                auto_create_session=True
            )
            message = types.Content(role="user", parts=[types.Part.from_text(text=text)])
            
            ctx.logger.info("-> Waking up The Planner Agent for Direct Chat...")
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

@planner_protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received ChatAck from {sender} for message {msg.acknowledged_msg_id}")

planner_uagent.include(planner_protocol, publish_manifest=True)

if __name__ == "__main__":
    planner_uagent.run()
