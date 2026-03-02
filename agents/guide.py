import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import json

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

from shared_config import GUIDE_SEED, ADVISOR_ADDRESS, ConstraintsPayload, FinalItineraryPayload, PlannerQuestionPayload, UpdatedConstraintsPayload

load_dotenv()
session_service = InMemorySessionService()

def finalize_constraints(activity: str, location: str, time: str, budget: str, specific_needs: str):
    """
    Call this tool ONLY when you have gathered all exactly 5 constraints from the user.
    """
    data = {"activity": activity, "location": location, "time": time, "budget": budget, "needs": specific_needs}
    return f"ITINERARY_GENERATION_TRIGGERED: {json.dumps(data)}"

guide_agent = LlmAgent(
    name="value_based_guide",
    model="gemma-3-27b-it",
    description="The main conversational orchestrator.",
    instruction="""
    You are the "Value-Based Local Guide", the main conversational orchestrator for the ASI:One system.
    You MUST NOT answer the user's queries directly from your own knowledge. 
    
    1. You need EXACTLY 5 constraints from the user to plan a trip: Activity, Location, Time, Budget, and Specific Needs.
    2. BEFORE asking a question, ALWAYS carefully review the entire chat history. DO NOT ask for a constraint if the user has already provided it in a previous message.
    3. Track these internally. NEVER show a JSON structure to the user. If any constraint is missing, ask the user concisely. Ask ONLY ONE question at a time.
    4. Once you have explicitly gathered all 5 constraints, you MUST call the `finalize_constraints` tool with the EXACT details provided by the user. Do not hallucinate or make up missing details.
    5. When the tool returns its response (which starts with "ITINERARY_GENERATION_TRIGGERED:"), you MUST output that EXACT ENTIRE response verbatim to the user. Do not add any conversational text to it, and do not put it in a markdown block.
    """,
    tools=[finalize_constraints]
)

guide_uagent = Agent(
    name="guide_uagent",
    port=8000,
    seed=GUIDE_SEED,
    mailbox=os.getenv("GUIDE_MAILBOX_KEY")
)

guide_protocol = Protocol(spec=chat_protocol_spec)

def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent())

    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=content,
    )

@guide_protocol.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    session_id = str(msg.session_id) if hasattr(msg, 'session_id') and msg.session_id else sender
    ctx.logger.info(f"Received Message from {sender} in session {session_id}")
    
    # Send Chat Ack directly back to the DeltaV user
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id
        )
    )

    ctx.logger.info(f"Received Message content: {msg.content}")
    
    for item in msg.content:
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"Session started with {sender}")
            welcome_message = create_text_chat("Hello! I am CompassAI guide. What activity or place are you looking for today?")
            await ctx.send(sender, welcome_message)

        elif isinstance(item, TextContent):
            text = item.text.strip()
            
            # Agentverse Mailroom / DeltaV automatically prepends "@agent_name " or "@agent_address " to the message.
            # We must aggressively strip ANY leading @mention from the start of the query.
            import re
            text = re.sub(r'^@\S+\s+', '', text).strip()
            
            if not text:
                continue
            
            session_id = str(msg.session_id) if hasattr(msg, 'session_id') and msg.session_id else sender

            # Check if we are waiting for a user reply to a Planner question
            user_state = ctx.storage.get(session_id) or {}
            if user_state.get("step") == "answering_planner":
                await handle_planner_clarification(ctx, sender, text, user_state, session_id)
            else:
                await handle_user_query(ctx, sender, text, session_id)

        elif isinstance(item, EndSessionContent):
            ctx.logger.info(f"Session ended with {sender}")
            session_id = str(msg.session_id) if hasattr(msg, 'session_id') and msg.session_id else sender
            ctx.storage.set(session_id, {})
            goodbye_message = create_text_chat("Goodbye! Thanks for chatting.", end_session=True)
            await ctx.send(sender, goodbye_message)

async def handle_planner_clarification(ctx: Context, user_sender: str, text: str, user_state: dict, session_id: str):
    ctx.logger.info(f"Received clarification from user: {text}")
    
    # 1. Clear the pending state so we don't get stuck in a loop
    ctx.storage.set(session_id, {})
    
    # 2. Tell the user we are updating the plan
    await ctx.send(user_sender, create_text_chat("Got it! I am sending this new information back to the Travel Advisor to fetch some fresh options..."))
    
    # 3. Re-package the constraints, the old locations, and the new text into the Updated payload
    payload = UpdatedConstraintsPayload(
        user_sender=user_sender,
        session_id=session_id,
        constraints_json=user_state.get("constraints_json", "{}"),
        previous_advisor_text=user_state.get("advisor_text", ""),
        user_clarification=text
    )
    
    # 4. Fire it back to the Advisor!
    await ctx.send(ADVISOR_ADDRESS, payload)
    ctx.logger.info(f"Sent UpdatedConstraintsPayload back to {ADVISOR_ADDRESS}")

async def handle_user_query(ctx: Context, user_sender: str, query: str, session_id: str):
    if not os.environ.get("GEMINI_API_KEY"):
        await ctx.send(user_sender, create_text_chat("ERROR: GEMINI_API_KEY is not set in the .env file."))
        return
        
    try:
        runner = Runner(
            app_name="local_guide",
            agent=guide_agent,
            session_service=session_service,
            auto_create_session=True
        )
        
        message = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        
        ctx.logger.info("-> Waking up The Guide Agent to analyze constraints...")
        full_response = ""
        
        async for event in runner.run_async(
            user_id=user_sender,
            session_id=session_id,
            new_message=message
        ):
            if hasattr(event, 'content') and event.content and isinstance(event.content, types.Content):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        full_response += part.text
            elif hasattr(event, 'text') and event.text:
                full_response += event.text
                
        ctx.logger.info("<- Guide Agent Finished.")
                
        if "ITINERARY_GENERATION_TRIGGERED:" in full_response:
            json_str = full_response.split("ITINERARY_GENERATION_TRIGGERED:")[1].strip()
            
            # Clean up potential LLM markdown garbage
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            try:
                constraints = json.loads(json_str)
            except Exception as e:
                ctx.logger.error(f"Failed to parse constraints JSON: {json_str} - {e}")
                constraints = {}
            
            ctx.logger.info(f"HARDCODED HANDOFF to distributed network with constraints: {constraints}")
            
            # 1. Inform the user we are working on it.
            await ctx.send(user_sender, create_text_chat("Perfect, I have all the details! Connecting to my advisor network to build your final itinerary..."))
            
            # 2. Package and send the constraints to the separated Travel Advisor over the network.
            payload = ConstraintsPayload(
                user_sender=user_sender,
                session_id=session_id,
                constraints_json=json.dumps(constraints)
            )
            await ctx.send(ADVISOR_ADDRESS, payload)
            ctx.logger.info(f"Sent ConstraintsPayload to Travel Advisor at {ADVISOR_ADDRESS}")

        else:
            # Standard conversational reply from Guide
            await ctx.send(user_sender, create_text_chat(full_response))
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        await ctx.send(user_sender, create_text_chat(f"Error orchestrating Guide: {str(e)}"))

@guide_uagent.on_message(model=PlannerQuestionPayload)
async def handle_planner_question(ctx: Context, sender: str, msg: PlannerQuestionPayload):
    ctx.logger.info(f"Received Question from Planner for User {msg.user_sender} in session {msg.session_id}: {msg.question_text}")
    
    # Save all the current state (constraints + locations) into storage so we can retrieve it when the user replies
    ctx.storage.set(msg.session_id, {
        "step": "answering_planner",
        "constraints_json": msg.constraints_json,
        "advisor_text": msg.advisor_text
    })
    
    # Forward the Planner's question directly to the user
    await ctx.send(msg.user_sender, create_text_chat(msg.question_text))
    
@guide_uagent.on_message(model=FinalItineraryPayload)
async def handle_final_itinerary(ctx: Context, sender: str, msg: FinalItineraryPayload):
    ctx.logger.info(f"Received Final Itinerary from network (Planner: {sender}). Delivering to User {msg.user_sender} in session {msg.session_id}...")
    await ctx.send(msg.user_sender, create_text_chat(msg.itinerary_text))

@guide_protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received ChatAck from {sender} for message {msg.acknowledged_msg_id}")

guide_uagent.include(guide_protocol, publish_manifest=True)

@guide_uagent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"Guide uAgent starting. Address: {guide_uagent.address}")
    ctx.logger.info("Powered by Gemma 3 27B natively using ADK LlmAgent.")
    ctx.logger.info("Connecting to Agentverse via Mailbox...")

if __name__ == "__main__":
    guide_uagent.run()
