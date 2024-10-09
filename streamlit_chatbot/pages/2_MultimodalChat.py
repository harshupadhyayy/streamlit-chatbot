import streamlit as st
from streamlit import session_state
import wikipedia
from pathlib import Path
import urllib.request
import qdrant_client
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from langchain_core.documents import Document
from llama_index.core.schema import ImageNode
from llama_index.core import StorageContext
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

data_path = Path("data_wiki")
image_path = Path("data_wiki")

if 'index' not in session_state:
    session_state.index = None

if 'custom_retriever' not in session_state:
    session_state.custom_retriever = None

if 'messages' not in session_state:
    session_state.messages = []


def retrieve_docs(query):
    client = qdrant_client.QdrantClient(
        url=st.secrets['QDRANT_CLUSTER_URL'],
        api_key=st.secrets['QDRANT_API_KEY'],)

    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection")
    image_store = QdrantVectorStore(
        client=client, collection_name="image_collection")
    
    index = MultiModalVectorStoreIndex.from_vector_store(vector_store=text_store, image_vector_store=image_store)

    retriever = index.as_retriever(
        similarity_top_k=3, image_similarity_top_k=5)
    matching_documents = retriever.retrieve(query)
    results_text = []
    results_images = []
    for res_node in matching_documents:
        if isinstance(res_node.node, ImageNode):
            # if res_node.score > 0.75:
            results_images.append(res_node.node.metadata["file_path"])
        else:
            results_text.append(Document(res_node.node.text))
    return results_text, results_images


# class CustomRetriever(BaseRetriever):
#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
#     ):
#         """Sync implementations for retriever."""
#         index = session_state.index
#         retriever = index.as_retriever(
#             similarity_top_k=3, image_similarity_top_k=5)
#         matching_documents = retriever.retrieve(query)
#         results_text = []
#         results_images = []
#         for res_node in matching_documents:
#             if isinstance(res_node.node, ImageNode):
#                 if res_node.score > 0.83:
#                     results_images.append(res_node.node.metadata["file_path"])
#             else:
#                 results_text.append(Document(res_node.node.text))
#         return results_text, results_images


def convert_name_to_slug(name):
    name = name.lower()
    name = name.replace(' ', '_')
    name = ''.join(char for char in name if char.isalnum() or char == '_')
    return name


def get_wikipedia_page(query):
    result = wikipedia.page(query)
    save_wikipedia_text(result.content, query)
    save_wikipedia_images(result.images, query)


def save_wikipedia_text(text, user_query):
    data_path = Path("data_wiki")
    name = convert_name_to_slug(user_query)
    with open(data_path / f"{name}.txt", "w") as fp:
        fp.write(text)


def save_wikipedia_images(image_list, user_query):
    image_uuid = 0
    MAX_IMAGES_PER_WIKI = 30

    user_query = convert_name_to_slug(user_query)

    images_per_wiki = 0
    try:
        for url in image_list:
            if url.endswith(".jpg") or url.endswith(".png"):
                image_uuid += 1
                image_file_name = user_query + "_" + url.split("/")[-1]
                print(image_file_name)
                urllib.request.urlretrieve(
                    url, image_path / f"{user_query}_{image_uuid}.jpg"
                )
                images_per_wiki += 1
                if images_per_wiki > MAX_IMAGES_PER_WIKI:
                    break
    except:
        print(str(Exception("No images found for Wikipedia page: ")) + user_query)


def create_index(user_query):
    client = qdrant_client.QdrantClient(
    url="https://6273b91d-acf0-47b9-92d3-e5fd8e7cf57c.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="enwTg98oHlFS_5LV4p1oYyW8wk-HFZFt1ij8yRticqk8gIgAki03CA",)

    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection"
    )

    image_store = QdrantVectorStore(
        client=client, collection_name="image_collection"
    )

    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    documents = SimpleDirectoryReader("./data_wiki/").load_data()
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    index.storage_context.persist("my_index")
    return index


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def process_retriever_output(retriever_output):
    text_docs, image_paths = retriever_output
    return {
        "context": format_docs(text_docs),
        "image_paths": image_paths
    }


def create_chain():
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """

    custom_runnable = RunnableLambda(retrieve_docs)

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model='gpt-4o')

    retriever_chain = RunnableParallel(
        retriever_output=custom_runnable,
        question=RunnablePassthrough()
    )

    process_chain = RunnableParallel(
        context=lambda x: process_retriever_output(
            x["retriever_output"])["context"],
        question=lambda x: x["question"],
        image_paths=lambda x: process_retriever_output(
            x["retriever_output"])["image_paths"]
    )

    answer_chain = RunnableParallel(
        answer=prompt | model | StrOutputParser(),
        image_paths=lambda x: x["image_paths"]
    )

    chain = retriever_chain | process_chain | answer_chain
    return chain


def get_response(message):
    chain = create_chain()
    response = chain.invoke(message)
    return response

def render_messages():
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('human'):
                st.write(message.content)


user_query = st.text_input("Enter the page title to find:")
submit_button = st.button("Submit")

if submit_button and user_query:
    with st.spinner("Processing..."):
        get_wikipedia_page(user_query)
        index = create_index(user_query=user_query)
        session_state['index'] = index
        print(session_state)
        st.success("Done")

render_messages()

query = st.chat_input("Ask anything")


if query:
    with st.chat_message('user'):
        st.write(query)

    human_prompt = HumanMessage(content=query)
    st.session_state.messages.append(human_prompt)

    response = get_response(query)

    with st.chat_message("assistant"):
        st.write(response['answer'])
        for image_path in response['image_paths']:
            st.image(image_path, width=200)

    st.session_state.messages.append(AIMessage(content=response['answer']))

