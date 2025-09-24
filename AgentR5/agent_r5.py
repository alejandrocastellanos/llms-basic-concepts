from dotenv import load_dotenv
load_dotenv()

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# 1. Cargar embeddings y base vectorial guardada
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_r5", embeddings, allow_dangerous_deserialization=True)

# 2. Crear retriever
retriever = vectorstore.as_retriever()

# 3. Definir modelo LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. Crear Prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Eres un asistente experto en Grupo R5, una fintech automotriz en Colombia.
Responde basándote únicamente en el contexto dado. Sé claro y detallado.

Contexto:
{context}

Pregunta: {question}
Respuesta:
"""
)

# 5. Crear el QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

# 6. Ejemplo de uso
while True:
    pregunta = input("\nHaz una pregunta sobre Grupo R5 (o escribe 'salir'): ")
    if pregunta.lower() in ["salir", "exit", "quit"]:
        break
    respuesta = qa_chain.run(pregunta)
    print("🤖:", respuesta)
