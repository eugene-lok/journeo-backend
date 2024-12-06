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

    Reply with a JSON object with both entities and a natural follow-up message:
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
        "clarificationMessage": "If there are no missing entities left, provide a natural, conversational follow-up question asking about missing information. Be concise but friendly. If all entities are fulfilled, return an empty string "
    }}

    Focus on extracting new information from: "{userInput}"
    """
)

def cleanJsonResponse(responseText: str) -> str:
    """Clean the LLM response by removing markdown code block markers."""
    cleaned = responseText.replace('```json', '').replace('```', '')
    return cleaned.strip()

def isEmptyValue(value: Any) -> bool:
    """Check if a value should be considered empty/missing."""
    if value is None:
        return True
    if isinstance(value, str):
        cleaned = value.lower().strip()
        return cleaned == "" or cleaned == "null"
    return False

def mergeEntities(previous: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge previous entities with newly extracted ones, preferring new values."""
    merged = previous.copy()
    for key, value in new.items():
        if value is not None:
            merged[key] = value
    return merged

def extractEntities(state: TravelPreferenceState, config):
    """Extract entities from user input and handle clarification if needed."""
    print("\n=== Starting Entity Extraction ===")
    
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
    
    print("User Input:", userInput)
    print("Previous Entities:", json.dumps(previousEntities, indent=2))
    
    # Format previous entities for the prompt
    previousEntitiesStr = json.dumps(previousEntities, indent=2) if previousEntities else "No previous entities"
    
    try:
        # Get LLM response
        response = llm.invoke(prompt)
        print("\nRaw LLM Response:", response.content)
        
        # Clean and parse JSON response
        cleanedResponse = cleanJsonResponse(response.content)
        print("\nCleaned Response:", cleanedResponse)
        
        responseData = json.loads(cleanedResponse)
        newEntities = responseData['entities']
        clarificationMessage = responseData['clarificationMessage']
        print("\nNew Entities:", json.dumps(newEntities, indent=2))
        
        # Merge with previous entities
        extractedEntities = mergeEntities(previousEntities, newEntities)
        print("\nMerged Entities:", json.dumps(extractedEntities, indent=2))
        
        # Determine missing entities - with explicit value checking
        missingEntities = []
        for key, value in extractedEntities.items():
            if isEmptyValue(value):
                missingEntities.append(key)
                print(f"Entity '{key}' is missing with value: {value}")
        
        print("\nMissing Entities:", missingEntities)
        print("Clarification Message:", clarificationMessage)
        print("\nIs Complete:", len(missingEntities) == 0)
        
        return {
            'userInput': userInput,
            'previousEntities': previousEntities,
            'extractedEntities': extractedEntities,
            'missingEntities': missingEntities,
            'clarificationMessage': clarificationMessage,
            'isComplete': len(missingEntities) == 0
        }
    except Exception as e:
        print(f"\nError in extractEntities: {str(e)}")
        
        # Fallback if parsing fails
        missingEntities = list(ENTITY_DESCRIPTIONS.keys())
        print("Falling back to default missing entities:", missingEntities)
        
        return {
            'userInput': userInput,
            'previousEntities': previousEntities,
            'extractedEntities': previousEntities,
            'missingEntities': missingEntities,
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