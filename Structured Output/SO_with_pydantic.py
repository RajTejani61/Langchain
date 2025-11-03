from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

class Review(BaseModel):
	key_themes : list[str] = Field(description="Write down all The key themes discussed in the review")
	summary : str = Field(description="The brief summary of the review")
	sentiment : Literal["Positive", "Negative", "Neutral"] = Field(description="Return sentiment of the review wither positive or negative or neutral")
	pros : Optional[list[str]] = Field(description="Write down all the pros from the review")
	cons : Optional[list[str]] = Field(description="Write down all the cons from the review")
	name : Optional[str] = Field(default=None, description="write the name of the reviewer")

input_txt =  """
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
sentiment='Positive'
key_themes=['Samsung Galaxy S24 Ultra', 'Performance', 'Camera', 'Battery Life', 'Price'] 
summary='The user is highly impressed with the Samsung Galaxy S24 Ultra, calling it a "powerhouse" due to its Snapdragon 8 Gen 3 processor, long-lasting 5000mAh battery, and stunning 200MP camera with impressive zoom. However, they note that the device is bulky, comes with bloatware, and has a steep $1,300 price tag.' 
pros=['Insanely powerful processor (great for gaming and productivity)', 'Stunning 200MP camera with incredible zoom capabilities', 'Long battery life with fast charging', 'S-Pen support is unique and useful'] 
cons=['Bulky and heavy-not great for one-handed use', 'Bloatware still exists in One UI', 'Expensive compared to compatitors'] 
name=None
"""