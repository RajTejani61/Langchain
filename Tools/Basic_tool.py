from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()


class WeatherInput(BaseModel):
    """Input for weather queries."""
    
    location: str = Field(description="City name or coordinates")
    
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str, include_forecast: bool) -> str:
    """Get current weather and optional forecast."""
    
    temp = 22 if units == "celsius" else 72
    
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result


model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

agent = create_agent(
	model=model,
	tools=[get_weather],
	system_prompt="You are a helpful assistant",
)

result = agent.invoke(
	input={"messages": [
        {"role": "user", "content": "what is the weather in sf in fahrenheit. Also include a 5 day forecast"}
    ]}
)
print(result) # The weather in San Francisco is 72 degrees Fahrenheit. The 5-day forecast is sunny