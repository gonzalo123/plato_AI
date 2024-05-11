import logging

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def get_vector_store_from_path(file_path):
    loader = TextLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    logger.info(f"Text divided in {len(all_splits)} splits")
    return Chroma.from_documents(
        documents=all_splits, embedding=GPT4AllEmbeddings()
    )


def get_chain(template, vector_store, llm):
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(dict(
        context=vector_store.as_retriever(),
        question=RunnablePassthrough(),
    ))
    return setup_and_retrieval | prompt | llm | output_parser
