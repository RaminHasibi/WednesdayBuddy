import os
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain, RetrievalQA
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers import PyPDFParser
from langchain.docstore.document import Document
from utils import extract_page_pdf
import chainlit as cl

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """Use the following pieces of context to answer the users question.
If you can not answer the question based on the given context, just say that you don't know, don't try to make up an answer.

And if the user greets with greetings like Hi, hello, How are you, etc reply accordingly as well.



Begin!
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


@cl.on_chat_start
async def on_chat_start():
    # Sending an action button within a chatbot message

    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    # Decode the file
    blob = Blob.from_data(file.content, path=os.path.join('pdf_files', file.path))

    pages = PyPDFParser().parse(blob)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


    # Split the text into chunks
    texts = text_splitter.split_documents(pages)

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_documents)(
        texts, embeddings
    )


    # Create a chain that uses the Chroma vector store
    chain = RetrievalQA.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        # prompt=prompt
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True)

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Pdf]

    for source_idx, source_doc in enumerate(source_documents):
        source_name = f"source_{source_idx}"
        # Create the text element referenced in the message

        text_elements.append(
            cl.Pdf(content=extract_page_pdf(source_doc.metadata['source'], source_doc.metadata['page']), name=source_name, display='page')
        )
    source_names = [text_el.name for text_el in text_elements]

    if source_names:
        answer += f"\nSources: {', '.join(source_names)}"
    else:
        answer += "\nNo sources found"
    await cl.Message(content=answer, elements=text_elements).send()