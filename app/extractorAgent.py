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
    missingEntities: List[str]
    clarificationMessage: str
    isComplete: bool

# Entity descriptions for clarification messages
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
    Your task is to maintain and update travel preference information, considering both previous and new information.

    CURRENT KNOWN INFORMATION:
    {previousEntities}

    NEW USER INPUT TO PROCESS: "{userInput}"

    INSTRUCTIONS:
    1. Extract new entities from the user input
    2. For any entity not mentioned in the new input, use null
    3. The final clarification message should be based on what's missing after combining:
       - Previously known entities (above)
       - Newly extracted entities (from current input)
    4. Do not ask about information that exists in either previous OR new entities

    Required entities:
    - Destinations (cities/countries/regions)
    - Budget
    - Trip Duration
    - Number of Travelers
    - Start Date
    - Traveling with Children
    - Traveling with Pets

    Example Scenario:
    If previous entities had "destinations": "London", and new input doesn't mention destinations,
    do NOT ask about destinations in the clarification message since it's already known.

    Reply with this JSON structure:
    {{
        "entities": {{
            "destinations": "extracted destination or null",
            "budget": "extracted budget or null",
            "duration": "extracted duration or null",
            "numTravellers": "extracted number or null",
            "startDate": "extracted date or null",
            "includesChildren": "true/false or null",
            "includesPets": "true/false or null"
        }},
        "clarificationMessage": "Ask ONLY about truly missing information (not found in either previous OR new entities). If ALL entities are accounted for when combining previous and new, return an empty string."
    }}
    """
)

def cleanJsonResponse(responseText: str) -> str:
    """Clean the LLM response by removing markdown code block markers."""
    cleaned = responseText.replace('```json', '').replace('```', '')
    return cleaned.strip()

def mergeEntities(previous: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge previous entities with newly extracted ones, preferring new values."""
    merged = {}
    # Include all keys from ENTITY_DESCRIPTIONS to ensure complete coverage
    for key in ENTITY_DESCRIPTIONS.keys():
        # Get value from new entities if it exists and isn't empty, otherwise from previous
        new_value = new.get(key)
        if not isEmptyValue(new_value):
            merged[key] = new_value
        else:
            prev_value = previous.get(key)
            if not isEmptyValue(prev_value):
                merged[key] = prev_value
            else:
                merged[key] = None
    return merged

def isEmptyValue(value: Any) -> bool:
    """Check if a value should be considered empty/missing."""
    if value is None:
        return True
    if isinstance(value, str):
        cleaned = value.lower().strip()
        return cleaned == "" or cleaned == "null" or cleaned == "none"
    return False

def extractEntities(state: TravelPreferenceState, config):
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
    previousEntitiesStr = json.dumps(previousEntities, indent=2) if previousEntities else "No previous entities"
    
    # Prepare prompt
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        userInput=userInput,
        previousEntities=previousEntitiesStr
    )
    
    try:
        # Get LLM response
        response = llm.invoke(prompt)
        
        # Clean and parse JSON response
        cleanedResponse = cleanJsonResponse(response.content)
        print("Response:", cleanedResponse)
        
        responseData = json.loads(cleanedResponse)
        newEntities = responseData['entities']
        clarificationMessage = responseData['clarificationMessage']
        print("Parsed new entities:", newEntities)
        
        # Merge with previous entities
        extractedEntities = mergeEntities(previousEntities, newEntities)
        print("Merged entities:", extractedEntities)
        
        # Determine missing entities
        missingEntities = [
            key for key in ENTITY_DESCRIPTIONS.keys()
            if isEmptyValue(extractedEntities.get(key))
        ]
        print("Missing entities:", missingEntities)
        
        return {
            'userInput': userInput,
            'previousEntities': previousEntities,
            'extractedEntities': extractedEntities,
            'missingEntities': missingEntities,
            'clarificationMessage': clarificationMessage,
            'isComplete': len(missingEntities) == 0
        }
    except Exception as e:
        print(f"Error in extractEntities: {str(e)}")
        return {
            'userInput': userInput,
            'previousEntities': previousEntities,
            'extractedEntities': previousEntities,
            'missingEntities': list(ENTITY_DESCRIPTIONS.keys()),
            'clarificationMessage': "I had trouble understanding that. Could you please provide some details about your trip?",
            'isComplete': False
        }

def createTravelPreferenceWorkflow():
    """Create LangGraph workflow for travel preference extraction."""
    workflow = StateGraph(TravelPreferenceState)
    workflow.add_node("extract", extractEntities)
    workflow.set_entry_point("extract")
    workflow.add_conditional_edges(
        "extract",
        lambda state: "complete" if state['isComplete'] else "incomplete",
        {
            "complete": END,
            "incomplete": END
        }
    )
    return workflow.compile(checkpointer=MemorySaver())

def main():
    app = createTravelPreferenceWorkflow()
    config = {"configurable": {"thread_id": "1"}}
    
    # First invocation
    initialInput = {
        'userInput': 'I want to visit Paris for a week',
        'previousEntities': None
    }
    result1 = app.invoke(initialInput, config=config)
    print("\nFirst invocation result:")
    print(json.dumps(result1, indent=2))
    
    # Second invocation with previous entities
    secondInput = {
        'userInput': 'I will be traveling with my wife and two kids on June 20. We are planning to spend $5000. I have no pets.',
        'previousEntities': result1['extractedEntities']
    }
    result2 = app.invoke(secondInput, config=config)
    print("\nSecond invocation result:")
    print(json.dumps(result2, indent=2))

if __name__ == "__main__":
    main()