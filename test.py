import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key =os.getenv("GROQ_API_KEY"),
    model = os.getenv("GROQ_MODEL_NAME")

)
response = llm.invoke("hi how r u doing")
print(response.content)