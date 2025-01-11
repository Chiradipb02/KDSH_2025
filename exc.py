import pathway as pw
import asyncio
import time

import io
import json
import logging
from typing import Callable, Coroutine, List, Tuple
from pathway.xpacks.llm.vector_store import VectorStoreServer
from sentence_transformers import SentenceTransformer

import PyPDF2

logger = logging.getLogger(__name__)

def splitter_function(text: str) -> list[tuple[str, dict]]:
    """
    Simple splitter that divides text into sentences.
    
    Args:
        text (str): The input text to split.
    
    Returns:
        list[tuple[str, dict]]: A list of tuples containing each sentence and an empty metadata dictionary.
    """
    sentences = text.split('. ')
    return [(sentence.strip(), {}) for sentence in sentences if sentence]

def pdf_parser(file_bytes: bytes) -> List[Tuple[str, dict]]:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
        
        # metadata = {
        #     "title": reader.metadata.title if reader.metadata.title else "Unknown",
        #     "number_of_pages": len(reader.pages),
        # }

        # append text to test.txt
        with open("test.txt", "a") as f:
            f.write(text)
        
        return [("data", {"data": text})]
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        return [("", {"error": str(e)})]

table = pw.io.gdrive.read(
    object_id="13eDgt0YghQU2qlogGrTrXJzfD0h0F2Iw",
    service_user_credentials_file="kdsh-pathway-72c63a387058.json",
    mode="streaming",
    with_metadata=True,
)
pw.io.jsonlines.write(table, "test.jsonl")
vector_store = VectorStoreServer(
    table,
    parser=pdf_parser,
    embedder=embed_function,
    splitter=splitter_function,
)
vector_store.run_server(host="127.0.0.1", port=8000, threaded=True, with_cache=False)
import time
time.sleep(10)
pw.run()