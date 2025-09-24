from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4o")

# chain 1
prompt_1 = PromptTemplate.from_template("Explica el concepto de {topic} de forma detallada")
chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="explanation")

# chain 2
prompt_2 = PromptTemplate.from_template("Resume lo siguiente en un parrafo: \n {explanation}")
chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="resumed_explanation")

# secuential chain
sequential_chain = SequentialChain(
    chains=[chain_1, chain_2],
    input_variables=["topic"],
    output_variables=["explanation", "resumed_explanation"]
)

result = sequential_chain.invoke({"topic": "Programacion orientada a objetos"})

print(result["resumed_explanation"])
