import redis
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from redis.commands.search.query import Query
from redisvl.query.filter import Tag, List
from openai import OpenAI
import time
import json
from config import OPENAI_API_KEY

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Load an embedding model
model_name = "sentence-transformers/all-mpnet-base-v2"

embedding_model = HFTextVectorizer(model_name)

# Initialize SemanticCache
semantic_cache = SemanticCache(name= "llm-cache", distance_threshold= 0.1, ttl= 300, redis_client= redis_client, vectorizer= embedding_model)

# simple performance test with llm
api_key = OPENAI_API_KEY
openai_client = OpenAI(api_key=api_key)
system_prompt = "You are an assistance to help users with their question and answers. Kindly refer to the questions and answer properly."

def get_response(query):
    start = time.time()
    if response := semantic_cache.check(prompt=query):
        print("Cache hit..")
        response_time = time.time() - start
        return response, response_time
    else:
        print("Cache miss..")
        response = openai_client.chat.completions.create(
            model= "gpt-4o",
            messages= [{
                        "role": "system",
                        "content": [{"type": "text", "text": f"{system_prompt}"}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"{query}"}],
                    }]
                    
        )
        response_time = time.time() - start
        
        # store the query and response in cache
        semantic_cache.store(
            prompt=query,
            response=response.choices[0].message.content,
        )
        return response.choices[0].message.content, response_time
        
# with cache miss
query = "how many people lived in srilanka in 2023"
response, response_time = get_response(query)
print(f"Response 1: {response}\n with Response time {response_time}")

# with cache hit
query = "what is the population of srilanka in 2023"
response, response_time = get_response(query)
print(f"Response 2: {response}\n with Response time {response_time}")


# clearing up old semantic cache 
semantic_cache.delete()
print("Semantic Cache has been deleted..")

############## Advanced queries with metadata ###################################################

# Re-initialize SemanticCache for advanced queries
semantic_cache = SemanticCache(name= "llm-cache", distance_threshold= 0.1, ttl= 300, redis_client= redis_client, vectorizer= embedding_model, filterable_fields=[{"name": "country", "type": "tag"}, {"name": "citizen_of", "type": "tag"}])

# storing visa info for Sri Lankan citizens
semantic_cache.store(
    prompt="Do I need a visa to travel?",
    response="Yes, Sri Lankan citizens require a visa to travel to United Kingdom.",
    filters={"country": "UK", "citizen_of": "Sri Lanka"}
)

semantic_cache.store(
    prompt="Do I need a visa to travel?",
    response="No, Sri Lankan citizens do not need a visa to enter Singapore for short visits.",
    filters={"country": "Singapore", "citizen_of": "Sri Lanka"}
)

# query with metadata: country = Singapore and citizen = Sri Lanka
filter_expression = (Tag("country") == "Singapore") & (Tag("citizen_of") == "Sri Lanka")

query = "Do I need a visa to travel?"

if response := semantic_cache.check(prompt=query, filter_expression= filter_expression):
    print("Cache hit..")
    print(f"Response with metadata filters: {response}")


#################### deletion ######################

index = redis_client.ft(b'llm-cache')

res= index.search(Query("*"))
print(f"Before deletion: {res}")

# deleting a specific entry with entry_id
print("Entry to delete: ", res.docs[0].id)
index.delete_document(res.docs[0].id)

res= index.search(Query("*"))
print(f"After deletion: {res}")

# cleanup: deletion of semantic cache
semantic_cache.delete()
print("Semantic Cache has been deleted..")