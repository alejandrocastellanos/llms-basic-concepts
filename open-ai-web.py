import datetime
import json
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.schema import SystemMessage

text_system_prompt = '''
Solo puedes responder preguntas sobre noticias virales actuales. 
Si el usuario solicita cualquier otra cosa, debes rechazar la petición educadamente. 
No respondas consultas que no sean sobre noticias virales.
'''

class ViralNewsAgent:

    def __init__(self, model: str = 'gpt-4o'):
        self._llm = ChatOpenAI(temperature=0, model=model)
        self._search_tool = TavilySearchResults()
        self._system_prompt = SystemMessage(content=text_system_prompt)
        self._agent = None

    def _initialize_agent(self):
        self._agent = initialize_agent(
            tools=[self._search_tool],
            llm=self._llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            system_message=self._system_prompt,
            handle_parsing_errors=True
        )

    @staticmethod
    def _final_response(response) -> dict:
        response = response.replace("```json", "").replace("```", "")
        return json.loads(response)

    def _get_viral_news_agent(self, topic) -> dict:
        today = datetime.datetime.now()
        structure = [{"link": "https://example.com", "title": "Noticia 1", "summary": "Resumen de la noticia 1",
                      "topic": "Tema de la noticia"}]
        query = (f'''
            Dame las 3 noticias virales de hoy del siguiente tema: {topic}. 
            En tu respuesta debes agregar el link de la noticia, un titulo llamativo y un resumen corto de la noticia. 
            Recuerda: solo responde sobre noticias virales. 
            Dame toda la respuesta en el mismo idioma de la pregunta. 
            Ten encuenta el año actual y el mes actual es {today.month} y el año es {today.year}. 
            La respuesta la debes dar en formato json con la siguiente estructura: 
            {json.dumps(structure)}            
        ''')
        response = self._agent.run(query)
        return self._final_response(response)

    def run_viral_news(self, topic):
        self._initialize_agent()
        response = self._get_viral_news_agent(topic)
        return response

viral_news_agent = ViralNewsAgent()

response = viral_news_agent.run_viral_news("colombia")
