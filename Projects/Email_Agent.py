from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()


class EmailState(MessagesState):
    draft_email: str | None
    human_feedback : str | None
    final_email: str | None

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


def generate_email(state:EmailState):
    
    human_feedback = state.get("feedback", "")
    draft_email = state.get("draft_email", "")
    
    system_prompt = f"""
    You are a professional sales assistant.
        - Write a concise, polite cold email.
        - Do NOT send the email.
        - Only generate a draft.
        - Professional tone.
        - if user gives feedback, use it to rewrite the previous draft
        - human_feedback : {human_feedback}
        - draft_email : {draft_email}
    """
    
    message = [
		SystemMessage(content=system_prompt),
		HumanMessage(content=state["messages"][-1].content)
	]
    response = llm.invoke(message)
    print("Generated email : \n", response.content)
    
    return Command(
		update={"draft_email": response.content},
		goto="human_approval"
	)

def human_approval(state:EmailState):
    
    human_decision = interrupt({
        "question" : "approve, reject, or give feedback for regeneration",
        # "draft_email": state["draft_email"],
        "expected_format" : {
            "decision" : "approve | reject | feedback",
            "feedback" : "only if decision == feedback"
        },
        })
    
    decision_type = human_decision.get("decision", "approve")
    
    if decision_type == "approve":
        return Command(
			update={"final_email" : state["draft_email"]},
			goto="send_email"
		)
    
    elif decision_type == "reject":
        print("\nEmail rejected by user. Workflow stopped.")
        return Command(goto=END)
    
    elif decision_type == "feedback":
        feedback = human_decision.get("feedback", "")
        print(f"\nUser feedback: {feedback}")
        return Command(
            update={
                "messages":[HumanMessage(content=f"Rewrite the email using this feedback: {feedback}")],
                "feedback": feedback
            },
			goto="generate_email"
		)
    
    return Command(goto=END)

def send_email(state:EmailState):
    print(f"\nSending email...")
    print(f"\nEmail Sent successfully...")
    # print(state["final_email"])
    
    return Command(
		update={"final_email" : state["final_email"]},
		goto=END,
	)

checkpointer = InMemorySaver()
graph = StateGraph(EmailState)

graph.add_node("generate_email", generate_email)
graph.add_node("human_approval", human_approval)
graph.add_node("send_email", send_email)

graph.add_edge(START, "generate_email")

app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    initial_state = {
        "messages" : [HumanMessage(content="generate a short and simple invitation email")]
    }
    
    state = app.invoke(initial_state, config = {"configurable": {"thread_id": "user-session-1"}}) # type: ignore
    

    if "__interrupt__" in state:
        interrupt_playload = state["__interrupt__"][0].value
        print(f"HITL : {interrupt_playload}")
        
        decision = input("Your decision : ").strip().lower()
        
        resume_data = {"decision": decision}
        
        if decision == "feedback":
            resume_data["feedback"] = input("Enter feedback: ")
        else:
            resume_data["decision"] = "feedback"
            resume_data["feedback"] = decision
        
        result = app.invoke(
            Command(resume=resume_data),
            config={"configurable": {"thread_id": "user-session-1"}}
        )
