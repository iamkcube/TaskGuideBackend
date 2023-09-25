from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from flask_cors import CORS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import uuid

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:5174",
     "https://taskguide-ciphersix.web.app", "https://taskguide-ciphersix.firebaseapp.com"])

os.environ["OPENAI_API_KEY"] = os.environ["OPEN_API_KEY"]


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    try:
        pdf_file = request.files['pdfFile']
        print(pdf_file)
        print(pdf_file.filename)
        if pdf_file:
            file_name: str = pdf_file.filename if pdf_file.filename else ".pdf"

            if os.path.exists(file_name):
                file_name = str(uuid.uuid1()) + file_name
            pdf_file.save(f'docs/{file_name}')
            print(os.listdir("docs"))

            return jsonify({'message': 'PDF file uploaded successfully'}), 200
        else:
            return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_pdf_text(pdf_docs):
    text = ""
    os.chdir("docs")
    for pdf in pdf_docs:
        pdf_reader = PdfReader(open(pdf, "rb"))
        for page in pdf_reader.pages:
            text += page.extract_text()
    os.chdir("..")
    print(text)
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
        raw_text = get_pdf_text(os.listdir("docs"))
        text_chunks = get_text_chunks(raw_text)
        app.vectorstore = get_vectorstore(text_chunks)
        app.conversation_chain = get_conversation_chain(app.vectorstore)

    response = app.conversation_chain({'question': user_question})

    return jsonify({"response": response['chat_history'][-1].content})


if __name__ == '__main__':
    app.run()
