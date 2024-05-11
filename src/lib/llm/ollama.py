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
