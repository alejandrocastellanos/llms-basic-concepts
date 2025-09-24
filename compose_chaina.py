from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()


llm = ChatOpenAI(temperature=0, model="gpt-4o")


# step 1: sumarize
sumarize_prompt = PromptTemplate.from_template("Resume este texto en 3 líneas: {texto}")
sumarize_chain = LLMChain(llm=llm, prompt=sumarize_prompt, output_key="resumed_text")

# step 2: translate
translate_prompt = PromptTemplate.from_template("Traduce este texto a inglés: {resumed_text}")
translate_chain = LLMChain(llm=llm, prompt=translate_prompt, output_key="translated_text")

# step 3: creative hastag
hashtag_prompt = PromptTemplate.from_template("Crea 3 hashtag para el siguiente texto: {translated_text}")
hashtag_chain = LLMChain(llm=llm, prompt=hashtag_prompt, output_key="hashtag")

# compose
multi_chain = SequentialChain(
    chains=[sumarize_chain, translate_chain, hashtag_chain],
    input_variables=["texto"],
    output_variables=["resumed_text", "translated_text", "hashtag"],
    verbose=True
)

input = """
LangChain es una biblioteca poderosa de Python para crear aplicaciones que aprovechan modelos de lenguaje como GPT. 
Permite crear flujos de trabajo complejos encadenando pasos, integrando fuentes externas y manteniendo memoria 
del contexto.
"""

output = multi_chain.invoke({"texto": input})

print(output)