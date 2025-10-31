from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

import os

# os.environ['HF_HOME'] = 'D:/huggingFace_cache'

llm = HuggingFacePipeline.from_model_id(
	model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
	task="text-generation",
	pipeline_kwargs=dict(
		temperature=0.5,
		max_new_tokens=100
	)
)

model = ChatHuggingFace(llm=llm)

res = model.invoke("Waht is capital of india")

print(res)

"""
Device set to use cpu
content='<|user|>\n Waht is capital of india </s>\n<|assistant|>\n The capital of India is New Delhi.' 
additional_kwargs={} response_metadata={} id='lc_run--acc28867-0adf-4b91-b078-96eba24d1f68-0'
"""