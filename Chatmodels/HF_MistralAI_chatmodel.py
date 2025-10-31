from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

res = model.invoke("Waht is capital of india")

print(res)

"""
content=' The capital of India is New Delhi. It is a city that holds great historical and political significance. It serves as the center of the Government of India, and is a major financial and cultural hub. The city is known for its rich history, diverse culture, and bustling streets. It is a blend of the old and the new, with ancient monuments like the Red Fort and the Qutub Minar standing alongside modern skyscrapers and shopping malls.' 
additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 96, 'prompt_tokens': 11, 'total_tokens': 107}, 'model_name': 'mistralai/Mistral-7B-Instruct-v0.3', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} 
id='lc_run--c927183d-8c0e-4bdf-b5ea-8df610fab870-0' usage_metadata={'input_tokens': 11, 'output_tokens': 96, 'total_tokens': 107}
"""