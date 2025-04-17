from langchain_redis import RedisSemanticCache
from langchain_core.globals import set_llm_cache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI
import time

from config import OPENAI_API_KEY

redis_url = "redis://localhost:6379"

model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
)

semantic_cache = RedisSemanticCache(
    embeddings=embeddings,
    redis_url= redis_url,
    distance_threshold=0.1,
    ttl = 300
)

set_llm_cache(semantic_cache)
print("Semantic cache initialized successfully!")


## testing with llm
system_prompt = "You are an assistance to help users with their question and answers. Kindly refer to the questions and answer properly."
api_key = OPENAI_API_KEY
openai_client = OpenAI(api_key=api_key)

def get_response(query):
    start = time.time()
    response = openai_client.invoke(query)
    response_time = time.time() - start
    return response, response_time

# without cache hit
query = "how many people lived in srilanka in 2023"
response, response_time = get_response(query)
print(f"Response 1: {response}\n with Response time {response_time}")

# with cache hit
query = "what is the population of srilanka in 2023"
response, response_time = get_response(query)
print(f"Response 2: {response}\n with Response time {response_time}")