import ast
import chainlit as cl
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@cl.on_chat_start
async def retrieve_docs():
    llm = OpenAI(model='gpt-3.5-turbo-instruct',
                 temperature=0)

    # QUERY PORTION
    arxiv_query = None

    # Wait for the user to ask an Arxiv question
    while arxiv_query is None:
        arxiv_query = await cl.AskUserMessage(
            content="Please enter a topic to begin!", timeout=15).send()
    query = arxiv_query['output']

    # ARXIV DOCS PORTION
    arxiv_docs = ArxivLoader(query=query, load_max_docs=1).load()
    await cl.Message(content=f"We found some useful results online for {query} "
                             f"Displaying them now!").send()

    # ARXIV DISPLAY PORTION
    arxiv_papers = [f"{doc.metadata}" for doc in arxiv_docs]
    data_dicts = [ast.literal_eval(entry) for entry in arxiv_papers]
    # Prepare arXiv results for display
    arxiv_papers = [
        f"Published: {entry['Published']} \n Title: {entry['Title']} \n Authors: {entry['Authors']} \n Summary: {entry['Summary'][:100]}... \n---\n"
        for entry in data_dicts]

    await cl.Message(content=f"{arxiv_papers}").send()

    await cl.Message(content=f"Downloading and chunking articles for {query} "
                             f"This operation can take a while!").send()

    # DB PORTION
    pdf_data = []
    for doc in arxiv_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.create_documents([doc.page_content])
        pdf_data.append(texts)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    db = Chroma.from_documents(pdf_data[0], embeddings)

    # CHAIN PORTION
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

    # Let the user know that the system is ready
    await cl.Message(content=f"Database creation for `{query}` complete. You can now ask questions!").send()

    cl.user_session.set("chain", chain)


@cl.on_message
async def retrieve_docs(message: cl.Message):
    question = message.content
    chain = cl.user_session.get("chain")
    database_results = await chain.acall({"query": question}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(database_results['result']).send()
