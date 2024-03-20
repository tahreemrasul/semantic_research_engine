from utils import *
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def main():
    query = "vision transformers"
    arxiv_docs = get_arxiv_results(query)
    display_arxiv_results(arxiv_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    db = Chroma(persist_directory=query, embedding_function=embeddings)
    ingest_docs(arxiv_docs, db)

    llm = OpenAI(model='gpt-3.5-turbo-instruct',
                 temperature=0)
    question = "how many and which benchmark datasets and tasks were compared for light weight transformer? what was the performance of the light weight transformer?"

    result = get_qa_chain_answers(llm, db, question)
    print(result)


main()

