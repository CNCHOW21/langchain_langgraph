# @Time    : 2025/8/4 21:31
# @Author  : liuzhou
# @File    : 3_official_rag_langchain.py
# @software: PyCharm

import getpass
import os

import dotenv
from langchain_chroma import Chroma

dotenv.load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://news.sina.com.cn/c/xl/2025-08-04/doc-infivrrt1723300.shtml",),
    # bs_kwargs=dict(
    #     parse_only=bs4.SoupStrainer(
    #         class_=("post-content", "post-title", "post-header")
    #     )
    # ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# example_messages = prompt.invoke(
#     {"context": "fill content", "question": "现代化的本质是什么?"}
# ).to_messages()

# print(example_messages)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("全球实现现代化的国家和地区人口有多少?"):
    print(chunk, end="", flush=True)

# print(rag_chain.invoke("全球实现现代化的国家和地区人口有多少?"))