from typing import TypedDict, Annotated, Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

class Review(TypedDict):
    key_themes : Annotated[list[str], "Write down all The key themes discussed in the review"]
    summary : Annotated[str, "The brief summary of the review"]
    sentiment : Annotated[Literal["Positive", "Negative", "Neutral"], "Return sentiment of the review wither positive or negative or neutral"] 
    pros : Annotated[Optional[list[str]], "Write down all the pros from the review"]
    cons : Annotated[Optional[list[str]], "Write down all the cons from the review"]




input_txt = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to compatitors
"""


structured_output = model.with_structured_output(Review)

result = structured_output.invoke(input_txt)

print(result)


"""
{'sentiment': 'Positive', 
'pros': ['Insanely powerful processor (great for gaming and productivity)', 'Stunning 200MP camera with incredible zoom capabilities', 'Long battery life with fast charging', 'S-Pen support is unique and useful'], 
'cons': ['Bulky and heavy-not great for one-handed use', 'Bloatware still exists in One UI', 'Expensive compared to compatitors'], 
'summary': 'The user is highly impressed with the Samsung Galaxy S24 Ultra, calling it a powerhouse due to its Snapdragon 8 Gen 3 processor, long-lasting 5000mAh battery, and stunning 200MP camera with impressive zoom. They also appreciate the S-Pen for note-taking. However, they find the phone bulky and uncomfortable for one-handed use, dislike the pre-installed bloatware in One UI, and consider the $1,300 price tag to be a significant drawback.', 
'key_themes': ['Performance', 'Camera', 'Battery Life', 'S-Pen', 'Design', 'Software', 'Price']}
"""