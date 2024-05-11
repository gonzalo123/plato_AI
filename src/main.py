import logging

from lib.llm.ollama import llm
from lib.utils import get_chain, get_vector_store_from_path
from settings import DOCUMENT_PATH

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level='INFO',
    datefmt='%d/%m/%Y %X')

logger = logging.getLogger(__name__)


def ask_question(chain, question):
    logger.info(f"QUESTION: {question}")
    response = chain.invoke(question)

    print(response)


def main(template, path, questions):
    vector_store = get_vector_store_from_path(path)
    chain = get_chain(
        template=template,
        vector_store=vector_store,
        llm=llm)
    for question in questions:
        ask_question(
            chain=chain,
            question=question
        )


if __name__ == "__main__":
    template = """
        Responde a la pregunta basándote solo en el contexto que te indico.
        Responde usando citas del texto.

        Contexto: {context}
        Pregunta: {question}
        """
    questions = (
        "¿Cuáles son las ideas generales del texto?",
        "¿Cuál es postura de Sócrates ante su inminente condena?",
        "¿Puedes enumerar los protagonistas de la trama?"
    )
    main(template=template, path=DOCUMENT_PATH, questions=questions)
