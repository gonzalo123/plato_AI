# Building a Plato's expert AI with LLaMA3 and LangChain

Today, I'm delving into the realm of AI. My aim is to construct an AI capable of engaging in conversation about a 
specific document. For this experiment, I've chosen Plato's 'Apology of Socrates.' My goal is to develop an expert 
on this text, allowing me to pose questions and receive insightful responses. Let's dive in.

First of all I need a LLaMA3 model locally on my computer (MBP M2 24GB). To do that we can use [Ollama]
(https://ollama.com/). It's pretty straightforward to do that on Mac. Just follow the instructions, do

> brew install ollama

and that's all. We can start the server.

> ollama start

Now we need the model. We're going LLaMA3. A 4.7 GB model that we can download using:

> ollama pull llama3

And that's all. Our server is up and running ready to receive requests. Now we're going to create our script. We can
use simple HTTP requests to interact with Ollama using postman, for example, but it's more simple to use a framework
to handle the communications. We're going to use [LangChain](https://www.langchain.com/).

IAs models has a limitations of the number of tokens that we can use as I/O parameters. Apology of Socrates is a
book. Not very big but big enough to overcome this limit so, we need to split it in chucks. Also, we need to convert
those chunks into a vector store to be able the model to understand it. LangChain provides us document loaders to
read the document and to create this vector store. In my example I'm using a Apology of Socrates in txt, so I'm
going to use a TextLoader, but there are different loaders for PDFs, S3, Dataframes and much more things available in
LangChain SDK. With this function I obtain the vector store from a path.

```python
import logging

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
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
```

Now we need a chain to ask question to oru model. With this function I obtain my chain.

```python
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

logger = logging.getLogger(__name__)


def get_chain(template, vector_store, llm):
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(dict(
        context=vector_store.as_retriever(),
        question=RunnablePassthrough(),
    ))
    return setup_and_retrieval | prompt | llm | output_parser
```

I'm using an Ollama llm model, running locally on my computer as I explain before. LangChain allows us to use
different llm models (Azure, OpenAI,...). We can use those models if we've an account (they aren't for free)

```python
from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
import logging
from settings import OLLAMA_MODEL

logger = logging.getLogger(__name__)

llm = Ollama(
    model=OLLAMA_MODEL,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
logger.info(f"Model {OLLAMA_MODEL} loaded")
```

With those functions I can build finally my script. As you can see I prepare a template telling to llm what I want
and the set of questions I'm going to ask the model. Our main function will first fetch the vector store (it
takes several seconds). After that will load the llm from the chain (takes time also). Then we iterate between
questions and print the llm's answer in the terminal.

```python
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
        Answer the question based only on the context I give you.
        Answer using quotes from the text.

        Context: {context}
        Question: {question}
        """
    questions = (
        "What are the general ideas of the text?"
        "What is Socrates' position regarding his imminent condemnation?"
        "Can you list the protagonists of the plot?"
    )
    main(template=template, path=DOCUMENT_PATH, questions=questions)
```

And that's all. We have a Plato expert to chat with about one specific context (in this case Apology of Socrates). 
However, for a production-grade project, it's crucial to store our vector data in a database to avoid repetitive generation.

Note: In my example the questions, template and Plato's book is in spanish. Plato's book public domain.