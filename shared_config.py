import os
from dotenv import load_dotenv
from uagents import Agent, Model

load_dotenv()

# --- Seeds ---
GUIDE_SEED = "guide_agent_secret_seed_1"
ADVISOR_SEED = "travel_advisor_secret_seed_1"
PLANNER_SEED = "planner_secret_seed_1"

# --- Dynamic Address Resolution ---
# We load the cryptographic addresses from the .env file.
# This prevents `shared_config.py` from instantiating dummy agents that hijack the logger names in the main scripts.
GUIDE_ADDRESS = os.getenv("GUIDE_ADDRESS")
ADVISOR_ADDRESS = os.getenv("ADVISOR_ADDRESS")
PLANNER_ADDRESS = os.getenv("PLANNER_ADDRESS")

# --- Communication Models ---
class ConstraintsPayload(Model):
    user_sender: str # The original DeltaV user address so we can route back
    session_id: str
    constraints_json: str
    
class UpdatedConstraintsPayload(Model):
    user_sender: str
    session_id: str
    constraints_json: str
    previous_advisor_text: str # Keep track of the original locations
    user_clarification: str # The new info from the user

class UserPromptPayload(Model):
    session_id: str
    text: str

class LocationOptionsPayload(Model):
    user_sender: str
    session_id: str
    constraints_json: str
    advisor_text: str

class PlannerQuestionPayload(Model):
    user_sender: str
    session_id: str
    constraints_json: str
    advisor_text: str # Keep track of the locations so we don't lose them
    question_text: str # The question to ask the user

class FinalItineraryPayload(Model):
    user_sender: str
    session_id: str
    itinerary_text: str
