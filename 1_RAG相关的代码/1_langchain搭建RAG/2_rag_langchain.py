# @Time    : 2025/8/4 17:45
# @Author  : liuzhou
# @File    : 2_load_pdf_vect.py
# @software: PyCharm

"""
1.加载pdf文档
2.切分文档
3.Faiss数据库存储

"""
import getpass
import os

from PyPDF2 import PdfReader
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = ChatOpenAI(model="gpt-4o-mini", api_key="hk-fohxu91000053642c0f1403add5f1054c0c3f82f41723161", base_url="https://api.openai-hk.com/v1")

def format_docs(txts):
    return "\n\n".join(txt for txt in txts)

def split_pdf():
    pdf_text = ""
    pdfReader = PdfReader("./data/浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf")
    # print(len(pdfReader.pages))
    # page_numbers.extend([1]*2)
    # print(page_numbers)
    for page in pdfReader.pages:
        pdf_text += page.extract_text()
    text_splitter  = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    splits = text_splitter.split_text(pdf_text)
    vectorstore = Chroma.from_texts(splits, embedding=OpenAIEmbeddings(api_key="hk-fohxu91000053642c0f1403add5f1054c0c3f82f41723161", base_url="https://api.openai-hk.com/v1"))

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    print("="*10)
    # rag_chain.invoke("What is Task Decomposition?")
    print(rag_chain.invoke("个金客户经理分为哪些职位等级？"))
    # for chunk in rag_chain.stream("个金客户经理分为哪些职位等级？"):
    #     print(chunk, end="", flush=True)

if __name__ == '__main__':
   split_pdf()