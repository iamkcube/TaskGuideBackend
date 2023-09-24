from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = os.environ["OPEN_API_KEY"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(open(pdf, "rb"))
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


@app.route('/api/question', methods=['POST'])
def handle_user_input():
    user_question = request.json.get('question')
    print(user_question)

    if not hasattr(app, 'vectorstore'):
        raw_text = get_pdf_text([r'docs/SIH.pdf'])
        text_chunks = get_text_chunks(raw_text)
        app.vectorstore = get_vectorstore(text_chunks)
        app.conversation_chain = get_conversation_chain(app.vectorstore)

    response = app.conversation_chain({'question': user_question})

    return jsonify({"response": response['chat_history'][-1].content})


if __name__ == '__main__':
    app.run()
