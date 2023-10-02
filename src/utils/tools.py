from typing import Optional

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

from src.utils.config import load_config


class IntroTool(BaseTool):
    name = "intro_tool"
    description = "tool to describe how to use the AI Tutor Agent"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        config = load_config()
        intro = config.intro
        return intro

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class SentenceCheckTool(BaseTool):
    name = "sentence_check_tool"
    description = "useful for when user inputs message in polish and wants to check if it is correct"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""

        msgs = StreamlitChatMessageHistory()
        # memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

        config = load_config()

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            chat_memory=msgs
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

        return qa_chain.run(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class VerbConjugationPractiseTool(BaseTool):
    name = "conjugation_practise_tool"
    description = "useful for when user wants to practise polish verbs conjugation"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""

        msgs = StreamlitChatMessageHistory()
        # memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

        config = load_config()

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            chat_memory=msgs
            # memory_key="chat_history",
            # return_messages=True,
        )

        verb_template = PromptTemplate(input_variables=["history", "input"], template=config.verb_template)

        llm = ChatOpenAI(model_name="gpt-4", temperature=0.1, streaming=True)
        qa_chain = ConversationChain(
            llm=llm,
            verbose=True,
            memory=memory,
            # combine_docs_chain_kwargs={"prompt": qa_template},
        )
        qa_chain.prompt = verb_template

        return qa_chain.run(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
