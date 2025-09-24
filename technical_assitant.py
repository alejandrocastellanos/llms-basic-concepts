from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

TEMPLATE = """
Quiero que actues como un desarrollador de software senior, que a liderado equipos tecnicos y tiene experiencia en 
arquitectura de software.

Tema: {topic}

Respuesta:  
1. Explicación sencilla
2. Ejemplo de uso en codigo
3. Errores comunes y como evitarlos

No te puedes salir de este dominio de conocimiento:

- Programacion orientada a objetos
- Programacion funcional
- Desarrollo web
- Base de datos
- Arquitectura de software 
- IA
- Machine Learning
- DevOps
"""

prompt = ChatPromptTemplate.from_template(TEMPLATE)
llm = ChatOpenAI(temperature=0, model="gpt-4o")
chain = prompt | llm

def assistant(topic: str) -> str:
    response = chain.invoke({"topic": topic})
    return response.content

if __name__ == "__main__":
    print(assistant(input("Tema: ")))
