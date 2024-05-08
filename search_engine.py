import chainlit as cl
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from literalai import LiteralClient
from dotenv import load_dotenv

load_dotenv()

client = LiteralClient()

# This will fetch the champion version, you can also pass a specific version
prompt = client.api.get_prompt(name="test_prompt")
prompt = prompt.to_langchain_chat_prompt_template()
prompt.input_variables = ["context", "question"]


@cl.on_chat_start
async def retrieve_docs():
    if cl.context.session.client_type == "copilot":
        llm = ChatOpenAI(model='gpt-3.5-turbo',
                         temperature=0)

        # QUERY PORTION
        query = None

        # Wait for the user to ask an Arxiv question
        while query is None:
            query = await cl.AskUserMessage(
                content="Please enter a topic to begin!", timeout=15).send()
        arxiv_query = query['output']

        # ARXIV DOCS PORTION
        arxiv_docs = ArxivLoader(query=arxiv_query, load_max_docs=1).load()
        # Prepare arXiv results for display
        arxiv_papers = [
            f"Published: {doc.metadata['Published']} \n Title: {doc.metadata['Title']} \n Authors: {doc.metadata['Authors']} \n Summary: {doc.metadata['Summary'][:50]}... \n---\n"
            for doc in arxiv_docs]

        # Trigger popup for arXiv results
        fn_arxiv = cl.CopilotFunction(name="showArxivResults", args={"results": "\n".join(arxiv_papers)})
        await fn_arxiv.acall()

        await cl.Message(content=f"We found some useful results online for `{arxiv_query}` "
                                 f"Displaying them in a popup!").send()

        await cl.Message(content=f"Downloading and chunking articles for `{arxiv_query}` "
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
        chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=db.as_retriever(),
                                            chain_type_kwargs={
                                                "verbose": True,
                                                "prompt": prompt
                                            }
                                            )

        # Let the user know that the system is ready
        await cl.Message(content=f"Database creation for `{arxiv_query}` complete. You can now ask questions!").send()

        cl.user_session.set("db", db)
        cl.user_session.set("chain", chain)


@cl.on_message
async def retrieve_docs(message: cl.Message):
    if cl.context.session.client_type == "copilot":
        question = message.content
        chain = cl.user_session.get("chain")
        db = cl.user_session.get("db")
        # Create a new instance of the callback handler for each invocation
        cb = client.langchain_callback()
        variables = {"context": db.as_retriever(search_kwargs={"k": 1}), "query": question}
        database_results = await chain.acall(variables,
                                             callbacks=[cb])
        results = [f"Question: {question} \n Answer: {database_results['result']}"]
        # Trigger popup for database results
        fn_db = cl.CopilotFunction(name="showDatabaseResults", args={"results": "\n".join(results)})
        await fn_db.acall()
        await cl.Message(content=f"We found some useful results from our database for your question: `{question}`"
                                 f"Displaying them in a popup!").send()

