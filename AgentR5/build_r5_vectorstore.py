import json

from dotenv import load_dotenv
load_dotenv()

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# 1. Definir la información de Grupo R5
grupo_r5 = {
    "nombre": "Grupo R5",
    "tipo": "Fintech automotriz",
    "pais": "Colombia",
    "descripcion": "R5 es una super app y plataforma digital que simplifica la vida de los conductores en Colombia, ofreciendo servicios relacionados con seguros, tecnomecánica y gestión vehicular.",
    "servicios": {
        "SOAT": {"descripcion": "Compra, cotización y descarga en línea del SOAT digital con descuentos autorizados por ley."},
        "Tecnomecanica": {"descripcion": "Agendamiento de revisiones en CDA aliados y recordatorios automáticos de vencimiento."},
        "Alertas": {"descripcion": "Notificaciones sobre vencimientos de SOAT, tecnomecánica, multas e impuestos."},
        "PicoYPlaca": {"descripcion": "Consulta personalizada de pico y placa según la ciudad."},
        "ZonaR5": {"descripcion": "Programa de beneficios y descuentos en productos y servicios relacionados con vehículos."},
        "HerramientasAdicionales": {"descripcion": "Contratos de compra-venta, consulta de precios de autos, alertas de mercado y otros servicios digitales."}
    },
    "plataforma": {
        "disponibilidad": ["App Store", "Google Play"],
        "usuarios": "+2 millones",
        "calificacion": "4.7/5 promedio",
        "caracteristicas": ["facilidad de uso", "rapidez", "seguridad en pagos"]
    },
    "confiabilidad": {
        "reputacion": ["Trustpilot", "Google Play", "App Store"],
        "opiniones": "Usuarios destacan rapidez, seguridad y utilidad de recordatorios y alertas."
    },
    "resumen": "R5 es una super app colombiana que centraliza la compra del SOAT, agendamiento de tecnomecánica, recordatorios de trámites, consulta de pico y placa y beneficios exclusivos en un solo lugar, con enfoque digital y seguro."
}

# 2. Convertir a texto
r5_text = json.dumps(grupo_r5, ensure_ascii=False, indent=2)

# 3. Crear documento
documents = [Document(page_content=r5_text, metadata={"fuente": "Grupo R5"})]

# 4. Crear embeddings y base vectorial
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# 5. Guardar en disco
vectorstore.save_local("faiss_r5")

print("✅ Base vectorial de Grupo R5 creada y guardada en 'faiss_r5/'")
