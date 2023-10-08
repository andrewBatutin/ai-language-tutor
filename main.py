import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate

from src.utils.config import load_config
from src.utils.parser import AITutorParser
from src.utils.prompts.ua.tutor_agent import agent_format_instructions, agent_prefix, agent_suffix
from src.utils.tools import IntroTool, PrzypadkiPractiseTool, SentenceCheckTool, VerbConjugationPractiseTool

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
        # memory_key="chat_history",
        # return_messages=True,
    )

    qa_template = PromptTemplate(input_variables=["history", "input"], template=config.qa_template)

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.1, streaming=True)
    qa_chain = ConversationChain(
        llm=llm,
        verbose=True,
        memory=memory,
        # combine_docs_chain_kwargs={"prompt": qa_template},
    )
    qa_chain.prompt = qa_template
    return qa_chain


def main():
    st.title("👩🏻‍🏫 AI Tutor")

    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    if "agent" not in st.session_state:
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

        config = load_config()

        llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)
        tools = load_tools(["llm-math"], llm=llm)
        tools.extend([SentenceCheckTool(), VerbConjugationPractiseTool(), IntroTool(), PrzypadkiPractiseTool()])
        # tools = [SentenceCheckTool()]
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            agent_kwargs={
                "prefix": agent_prefix,
                "suffix": agent_suffix,
                "format_instructions": agent_format_instructions,
                "output_parser": AITutorParser(),
            },
        )
        st.session_state["agent"] = agent

    agent = st.session_state.agent

    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(prompt, callbacks=[st_callback])
            st.write(response)


if __name__ == "__main__":
    main()
