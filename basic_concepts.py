from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(temperature=0, model="gpt-4o")

prompt = ChatPromptTemplate.from_template("Give me 3 ideas based on {industry}")
chain = prompt | llm

result = chain.invoke({"industry":"insurance"})

print(result.content)
