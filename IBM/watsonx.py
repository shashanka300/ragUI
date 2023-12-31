import os
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain.llms import WatsonxLLM


load_dotenv()
my_api_key = os.getenv("WATSONX_APIKEY", None)
# my_api_endpoint = os.getenv("IBM_CLOUD_URL", None)
my_project_id = os.getenv("WATSONX_ID", None)


parameters = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1,
}



watsonx_llm = WatsonxLLM(
    model_id="meta-llama/llama-2-70b-chat",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=my_project_id,
    apikey=my_api_key,
    params=parameters,
)

print(watsonx_llm("Who is man's best friend?"))