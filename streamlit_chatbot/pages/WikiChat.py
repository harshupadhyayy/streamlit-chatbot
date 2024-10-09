import streamlit as st
import wikipedia
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from streamlit import session_state as ss
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever


def search_wikipedia(title: str):
    search_result = wikipedia.page(title)
    return search_result.content

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, title):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(f"faiss_index-{title}")
    ss.vector_store = vector_store

def render_messages():
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('human'):
                st.write(message.content)

def get_history_aware_retriever():
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    new_db = FAISS.load_local('faiss_index-virat', embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    model = ChatOpenAI(model='gpt-4o', temperature=0.3)

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def get_conversational_chain():
    model = ChatOpenAI(model='gpt-4o', temperature=0.3)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever = get_history_aware_retriever()

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_response(human_prompt):
    chain = get_conversational_chain()
    chain = chain.pick("answer")
    return chain.invoke({"input": human_prompt.content, "chat_history": st.session_state.messages})

if 'vector_store' not in ss:
    ss.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello I am a bot. How can I help you?")
    ]

page_title = st.text_input("Enter the page title to find: ")
submit_button = st.button("Submit")

if submit_button and page_title:
    with st.spinner("Processing..."):
        content = search_wikipedia(page_title)
        chunked_data = get_text_chunks(content)
        get_vector_store(chunked_data, "virat")
        st.success("Done")

st.divider()

render_messages()

query = st.chat_input("Ask anything")

if query:
    with st.chat_message('user'):
        st.write(query)

    human_prompt = HumanMessage(content=query)
    st.session_state.messages.append(human_prompt)

    with st.chat_message("assistant"):
        response = get_response(human_prompt)
        st.write(response)
    st.session_state.messages.append(AIMessage(content=response))

