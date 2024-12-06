import os
from typing import Annotated, Any, List, Dict, Type, Optional
from typing_extensions import TypedDict
import json
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

# Define State for the Agent
class TravelPreferenceState(TypedDict):
    userInput: str
    previousEntities: Optional[Dict[str, Any]]
    extractedEntities: Dict[str, Any]
    missing_entities: List[str]
    clarificationMessage: str
    IsComplete: bool

# Entity descriptions for generating clarification messages
ENTITY_DESCRIPTIONS = {
    'destinations': 'Where would you like to travel?',
    'budget': 'What is your budget for this trip?',
    'duration': 'How long do you plan to travel?',
    'numTravellers': 'How many people are traveling?',
    'startDate': 'When do you want to start your trip?',
    'includesChildren': 'Are you traveling with children?',
    'includesPets': 'Are you traveling with pets?'
}

# Entity extraction system prompt
ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["userInput", "previousEntities"],
    template="""
    Extract travel preference entities from the user's latest input. Some entities may have been previously provided.

    Previously extracted entities:
    {previousEntities}

    Extract or update the following entities from the new input:
    - Destinations (cities/countries/regions)
    - Budget
    - Trip Duration
    - Number of Travelers
    - Start Date
    - Traveling with Children
    - Traveling with Pets

    Reply with a JSON object with the following format:
    {{
        "destinations": "extracted destination or null",
        "budget": "extracted budget or null",
        "duration": "extracted duration or null",
        "numTravellers": "extracted number or null",
        "startDate": "extracted date or null",
        "includesChildren": "true/false or null",
        "includesPets": "true/false or null"
    }}

    Focus on extracting new information from: "{userInput}"
    """
)

def generateclarificationMessage(missing_entities: List[str]) -> str:
    """Generate a human-readable clarification message."""
    questions = [ENTITY_DESCRIPTIONS.get(entity, entity) for entity in missing_entities]
    return f"I need a bit more information. Could you please provide details about: {', '.join(questions)}?"

def merge_entities(previous: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge previous entities with newly extracted ones, preferring new values when available."""
    merged = previous.copy()
    for key, value in new.items():
        if value is not None:  # Only update if new value is not None
            merged[key] = value
    return merged

def clean_json_response(response_text: str) -> str:
    """Clean the LLM response by removing markdown code block markers."""
    # Remove ```json and ``` markers
    cleaned = response_text.replace('```json', '').replace('```', '')
    # Strip whitespace
    cleaned = cleaned.strip()
    return cleaned

def is_empty_value(value: Any) -> bool:
    """Check if a value should be considered empty/missing."""
    if value is None:
        return True
    if isinstance(value, str):
        # Check for empty string, "null", or just whitespace
        cleaned = value.lower().strip()
        return cleaned == "" or cleaned == "null"
    return False

def extract_entities(state: TravelPreferenceState, config):
    """Extract entities from user input and handle clarification if needed."""
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=1000,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    # Get user input and previous entities from state
    userInput = state['userInput']
    previousEntities = state.get('previousEntities', {}) or {}
    
    # Format previous entities for the prompt
    previousEntities_str = json.dumps(previousEntities, indent=2) if previousEntities else "No previous entities"
    
    # Prepare prompt
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        userInput=userInput,
        previousEntities=previousEntities_str
    )
    
    try:
        # Get LLM response
        response = llm.invoke(prompt)
        
        # Debug print
        print("Raw LLM Response:", response.content)
        
        # Clean and parse JSON response
        cleaned_response = clean_json_response(response.content)
        print("Cleaned Response:", cleaned_response)
        
        new_entities = json.loads(cleaned_response)
        print("Parsed new entities:", new_entities)
        
        # Merge with previous entities
        extractedEntities = merge_entities(previousEntities, new_entities)
        print("Merged entities:", extractedEntities)
        
        # Determine missing entities using the new checker
        missing_entities = [
            key for key, value in extractedEntities.items() 
            if is_empty_value(value)
        ]
        print("Missing entities:", missing_entities)
        
        # Generate clarification message if needed
        clarificationMessage = generateclarificationMessage(missing_entities) if missing_entities else ""
        
        return {
            'userInput': userInput,
            'previousEntities': previousEntities,
            'extractedEntities': extractedEntities,
            'missing_entities': missing_entities,
            'clarificationMessage': clarificationMessage,
            'IsComplete': len(missing_entities) == 0
        }
    except Exception as e:
        # Print error for debugging
        print(f"Error in extract_entities: {str(e)}")
        
        # Fallback if parsing fails
        missing_entities = list(ENTITY_DESCRIPTIONS.keys())
        return {
            'userInput': userInput,
            'previousEntities': previousEntities,
            'extractedEntities': previousEntities,
            'missing_entities': missing_entities,
            'clarificationMessage': generateclarificationMessage(missing_entities),
            'IsComplete': False
        }

def create_travel_preference_workflow():
    """Create LangGraph workflow for travel preference extraction."""
    workflow = StateGraph(TravelPreferenceState)
    
    # Extraction node
    workflow.add_node("extract", extract_entities)
    
    # Entry point
    workflow.set_entry_point("extract")
    
    # Add both conditional end nodes from extract
    workflow.add_conditional_edges(
        "extract",
        # Condition function determines which edge to take
        lambda state: "complete" if state['IsComplete'] else "incomplete",
        {
            # If complete, go to END
            "complete": END,
            # If incomplete, go to END with clarification
            "incomplete": END
        }
    )
    
    return workflow.compile(checkpointer=MemorySaver())

def main():

    app = create_travel_preference_workflow()
    
    # Config 
    config = {"configurable": {"thread_id": "1"}}
    
    # Test
    initial_input = {
        'userInput': 'I want to visit Paris for a week',
        'previousEntities': None
    }
    result1 = app.invoke(initial_input, config=config)
    print("\nFirst invocation result:")
    print(json.dumps(result1, indent=2))
    
    # Second invocation with previous entities
    second_input = {
        'userInput': 'I will be traveling with my wife and two kids',
        'previousEntities': result1['extractedEntities']
    }
    result2 = app.invoke(second_input, config=config)
    print("\nSecond invocation result:")
    print(json.dumps(result2, indent=2))

if __name__ == "__main__":
    main()