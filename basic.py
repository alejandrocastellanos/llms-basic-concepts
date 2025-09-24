# pip install langchain openai
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. Texto de ejemplo (contexto fijo)
documento = """
LangChain es un framework para construir aplicaciones impulsadas por modelos de lenguaje (LLMs).
Proporciona componentes para encadenar pasos como prompts, llamadas a LLMs, memoria y herramientas.
LangChain facilita trabajar con LLMs en tareas como chatbots, preguntas y respuestas, y análisis de datos.
"""

# 2. Definir el modelo
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 3. Crear un prompt template
plantilla = PromptTemplate(
    input_variables=["contexto", "pregunta"],
    template="""
Eres un asistente que responde preguntas basadas en el siguiente contexto:

{contexto}

Pregunta: {pregunta}
Respuesta:"""
)

# 4. Crear el chain
qa_chain = LLMChain(llm=llm, prompt=plantilla)

# 5. Hacer una pregunta
pregunta = "¿Para qué se utiliza principalmente LangChain?"
respuesta = qa_chain.run(contexto=documento, pregunta=pregunta)

print("Pregunta:", pregunta)
print("Respuesta:", respuesta)
