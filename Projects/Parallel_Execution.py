from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage

from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
load_dotenv()


class ResearchState(MessagesState):
    main_query: str 
    sub_questions: List[str]    
    research_notes: List[str]
    final_report: str


class SubQuestions(BaseModel):
    """Planning schema"""
    sub_questions: List[str] = Field(
        description="4–6 focused research sub-questions"
    )

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.3
)

def supervisor_agent(state: ResearchState):
    planner = llm.with_structured_output(SubQuestions)

    plan = planner.invoke(
        f"""
        Break the following question into 4–6 research sub-questions.
        
        Question:
        {state['main_query']}
        """
    )
    
    return {
        "sub_questions": plan.sub_questions, # type: ignore
    }


def sub_query_research_agent(sub_query: str) -> str:
    """
    Research a single sub-question.
    """
    prompt = f"""
    You are a research agent.
    Provide factual findings in 4–5 sentences.
    
    Topic: {sub_query}
    """
    
    response = llm.invoke(prompt)
    
    return f"Topic: {sub_query} \n {response.content}"


def run_research(state: ResearchState):
    notes = []
    
    for q in state["sub_questions"]:
        result = sub_query_research_agent(q)
        notes.append(result)
    
    return {"messages": notes}


def create_final_report(state: ResearchState):
    
    prompt = f"""
    Write a structured research report.
    
    Main Question:
    {state['main_query']}
    
    Research Findings:
    {'\n\n'.join(state["research_notes"])}
    """
    
    final = llm.invoke(prompt)
    
    return {
        "final_report": final.content
    }


graph = StateGraph(ResearchState)

graph.add_node("planner", supervisor_agent)
graph.add_node("tools", run_research)
graph.add_node("final_report", create_final_report)

graph.add_edge(START, "planner")
graph.add_edge("planner", "tools")
graph.add_edge("tools", "final_report")
graph.add_edge("final_report", END)

app =  graph.compile()

query = "What are the benefits and challenges of artificial intelligence in healthcare?"

for response in app.stream({
    "messages": [HumanMessage(content=query)],
    "main_query": query,
    "sub_questions": [],
    "research_notes": [],
    "final_report": ""
}):
    print("\n===== RESPONSE =====")
    print(response)