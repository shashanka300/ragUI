import os
from dotenv import load_dotenv
import genai.extensions.langchain
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai import Credentials, Model, PromptPattern
load_dotenv() 

my_api_key = os.getenv("GENAI_KEY", None)
my_api_endpoint = os.getenv("GENAI_API", None)

# creds object
creds = Credentials(api_key=my_api_key, api_endpoint=my_api_endpoint)

params = GenerateParams(decoding_method="greedy",
            max_new_tokens=1536,
            min_new_tokens=0,
            repetition_penalty=1,
            stop_sequences = ["]"])


# As LangChain Model
langchain_model = LangChainInterface(model='thebloke/mixtral-8x7b-instruct-v0-1-gptq', params=params, credentials=creds)

def U_bam(prompt):
    result = langchain_model(prompt)
    return result

