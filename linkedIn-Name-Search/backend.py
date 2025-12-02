from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
import json


class Middleware:
    """Middleware to process and fromat results of agent"""
    
    @staticmethod
    def filter_results(results, min_score=0.3):
        """Filter results based on relevance score"""
        
        return [result for result in results if result['score'] >= min_score]
    
    @staticmethod
    def format_profiles(raw_results):
        """Format raw search results into structured LinkedIn profiles"""
        
        if not raw_results:
            return []
        
        profiles = []
        for result in raw_results:
            profile = {
				'name' : result.get('title', 'Unknown'),
				'url' : result.get('url', ''),
				'snippet' : result.get('content', ''),
				'score' : result.get('score', 0),
			}
            profiles.append(profile)
        return profiles
    

class LinkedInSearchAgent:
    """Agent for searcing LinkedIn profiles"""
    
    def __init__(self, google_api_key, tavily_api_key):
        self.google_api_key = google_api_key
        self.tavily_api_key = tavily_api_key
        self.middleware = Middleware()
        self.agent_executor = None
        
    def _search_profiles(self, query_params: dict):	
        
        """
        Search LinkedIn profiles/comapanies using Tavily API
        query_params format : "name|city|role|college|company" (city, role, college and company are optional)
        """
        
        try :
			# Parse query parameters
            if isinstance(query_params, str):
                try:
                    query_params = json.loads(query_params)
                except json.JSONDecodeError:
                    return "Invalid JSON format passed to LinkedInSearch tool."
            
            search_type = query_params.get("type", "person")   # person / company
            name = query_params.get("name")
            city = query_params.get("city")
            role = query_params.get("role")
            college = query_params.get("college")
            company = query_params.get("company")

            if not name:
                return "Error: 'name' is required."
            
            query = f'site:linkedin.com/in/ "{name}"'
            
            if city:
                query += f' {city}'
            if role:
                query += f' {role}'
            if college:
                query += f' "{college}"'
            if company:
                query += f' "{company}"'
            
            # Initialize tavily client
            client = TavilyClient(api_key=self.tavily_api_key)
            
            response = client.search(
				query=query,
				search_depth="advanced",
				max_results=5,
				include_domains=["linkedin.com/in", "linkedin.com/company"]
			)
            
            raw_results = response.get('results', [])
            filtered_results = self.middleware.filter_results(raw_results)
            fromatted_profiles = self.middleware.format_profiles(filtered_results)
            
            if not fromatted_profiles:
                return f"No linkedin profiles found for {name}" + (f"in {city}" if city else "") + (f" in {role}" if role else "")
            
            output = f"Found {len(fromatted_profiles)} LinkedIn profiles : \n\n"
            for idx, profile in enumerate(fromatted_profiles, 1):
                output += f"{idx}. {profile['name']} \n"
                output += f"\tURL: {profile['url']} \n"
                output += f"\tInfo: {profile['snippet'][:200]}...\n"
                output += f"\tRelevance: {profile['score']:.2f}\n\n"
                
            return output
            
        except Exception as e:
            return f"Error Searching LinkedIn profiles: {str(e)}"
        
    
    def _create_agent(self):
        """Create Lnaghcin agent with LinkedIn search tool"""
        
        # Create Tool 
        linkedin_search_tool = Tool(
			name="LinkedInSearch",
			func=self._search_profiles,
			description="""
            Search for LinkedIn profiles using structured JSON input.

			Required:
				- type: "person" or "company"
				- name: person's name or company name

			Optional filters:
				- city
				- role
				- college
				- company

			Examples:
			{{
				"type": "person",
				"name": "Rohan Shah",
				"city": "Mumbai",
				"college": "IIT Bombay"
			}}

			{{
				"type": "company",
				"name": "Infosys"
			}}
			
			""",
		)
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
			model="gemini-2.5-flash",
			google_api_key=self.google_api_key,
			temperature=0)
        
        template = """
        You are a LinkedIn search assistant. Your job is to help users find LinkedIn profiles or companies.
        
        You have access to the following tool: {tools}
        Tool Names : {tool_names}
        
        When a user asks to search for someone, use the LinkedInSearch tool with the format: type|name|city|role|college|company
        When searching, ALWAYS call the LinkedInSearch tool using JSON input.
        Examples:
		- For a person: {{"type": "person", "name": "John Doe", "city": "New York"}}
		- For a person with role: {{"type": "person", "name": "Amit Shah", "role": "Data Engineer"}}
		- For a college search: {{"type": "person", "college": "MIT"}}
		- For a company: {{"type": "company", "name": "Google"}}

		Use the following format:
		Question : the input question you must answer
		Thought : you should always think about what to do
		Action : Use one of the tools to get the answer
		Action Input: the input to the action
		Observation: the result of the action
		... (this Thought/Action/Action Input/Observation can repeat N times)
		Final Answer: the final answer to the original input question

		Question : {input}
		Thought : {agent_scratchpad}
        """
        
        prompt = PromptTemplate(
			template=template,
			input_variables=["input", "agent_scratchpad"],
			partial_variables={
				"tools": linkedin_search_tool.description,
				"tool_names": linkedin_search_tool.name
			}
		)
        
        agent = create_react_agent(
			llm=llm,
			tools=[linkedin_search_tool],
			prompt=prompt
		)
        
        self.agent_executor = AgentExecutor(
			agent=agent,
			tools=[linkedin_search_tool],
			verbose=True,
			handle_parsing_errors=True
		)
        
        return self.agent_executor
    
    def search(self, name, city=None, role=None, college=None, company=None, type="person"):
        """
        Main search method to be called from frontend
        
        Args:
            name (str): Person's name to search
            city (str, optional): City filter
            role (str, optional): Role/position filter
            college (str, optional): College filter
			company (str, optional): Company filter
			type (str, optional): Type of search (person or company)
            
        Returns:
            dict: Search results with status and output
        """
        
        try:
            # create agent if not exists
            if not self.agent_executor:
                self._create_agent()
            
            # create query
            payload = {
				"type": type,
				"name": name,
				"city": city,
				"role": role,
				"college": college,
				"company": company
			}
            
            query = f"Search LinkedIn with info: {payload}"
            # Execute search
            result = self.agent_executor.invoke({"input": query}) # type: ignore
            
            return {
				"status" : "Success",
				"output": result
			}
            
        except Exception as e:
            return {
				"status" : "Error",
				"output": f"Error Searching LinkedIn profiles: {str(e)}"
			}


# from dotenv import load_dotenv
# load_dotenv()
# import os

# GPPGLE_API = os.getenv("GOOGLE_API_KEY")
# TAVILY_API = os.getenv("TAVILY_API_KEY")
# if __name__ == "__main__":
# 	agent = LinkedInSearchAgent(google_api_key=GPPGLE_API, tavily_api_key=TAVILY_API)
# 	result = agent.search(type="person", name="raj tejani", city="Surat", college="SSASIT")
# 	# print(result)

