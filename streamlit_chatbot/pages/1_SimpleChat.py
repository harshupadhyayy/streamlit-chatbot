import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_response():
    prompt = st.session_state.prompt
    model = st.session_state.model
    chat_history = st.session_state.messages
    chain = prompt | model
    return chain.stream({"chat_history": chat_history})

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


st.set_page_config(
    page_title="Simple Chat",
    page_icon="👋",
)


st.title("Simple Chat")




render_messages()

query = st.chat_input("Ask anything")

if query:
    with st.chat_message('user'):
        st.write(query)

    human_prompt = HumanMessage(content=query)
    st.session_state.messages.append(human_prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(get_response())

    st.session_state.messages.append(AIMessage(content=response))
