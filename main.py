import os

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from src.utils.config import load_config

load_dotenv()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)

@st.cache_resource
def conversational_chain():
    config = load_config()

    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(
        #memory_key="chat_history",
        #return_messages=True,
    )

    qa_template = PromptTemplate(input_variables=["history", "input"], template=config.qa_template)

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.1, streaming=True)
    qa_chain = ConversationChain(
        llm=llm,
        verbose=True,
        memory=memory,
        #combine_docs_chain_kwargs={"prompt": qa_template},
    )
    qa_chain.prompt = qa_template
    return qa_chain


def main():
    st.title("👩🏻‍🏫 AI Tutor")

    if "qa_chain" not in st.session_state:
        st.session_state["qa_chain"] = conversational_chain()
    qa_chain = st.session_state.qa_chain

    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(placeholder="Ask me anything!")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[stream_handler])
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
