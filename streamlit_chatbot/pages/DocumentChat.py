import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



def get_conversational_chain():
    new_db = st.session_state.vector_store
    retriever = new_db.as_retriever()

    model = ChatOpenAI(temperature=0.3)
    prompt = hub.pull("rlm/rag-prompt")

    return (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_response(human_prompt):
    chain = get_conversational_chain()
    return chain.stream(human_prompt.content)


def render_messages():
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('human'):
                st.write(message.content)


if 'model' not in st.session_state:
    st.session_state.model = ChatOpenAI(api_key=st.secrets['OPENAI_API_KEY'])

if 'prompt' not in st.session_state:
    st.session_state.prompt = ChatPromptTemplate(
        [
            SystemMessage(
                "You are helpful assistant. Answer all the question to your best ability. If you don't know the answer to anything, then convey the same to the user"),
            MessagesPlaceholder(variable_name="chat_history")
        ]
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello I am a bot. How can I help you?")
    ]

if 'pdf_docs' not in st.session_state:
    st.session_state.pdf_docs = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

st.set_page_config(
    page_title="Document Chat",
    page_icon="👋",
)


st.title("Document Chat")


render_messages()

query = st.chat_input("Ask anything")

if query:
    with st.chat_message('user'):
        st.write(query)

    human_prompt = HumanMessage(content=query)
    st.session_state.messages.append(human_prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(get_response(human_prompt))

    st.session_state.messages.append(AIMessage(content=response))
