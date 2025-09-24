import os
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain_openai import ChatOpenAI

load_dotenv()


llm = ChatOpenAI(temperature=0.7, model="gpt-4o")

memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
)

while True:
    print("👨‍💻 Asistente Técnico LangChain (escribe 'salir' para terminar)\n")
    user_input = input("Tú: ")
    if user_input.lower() == "salir":
        break
    response = conversation.invoke({"input": user_input})
    print("🤖:", response["response"])
