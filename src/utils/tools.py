from typing import Optional

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

from src.utils.config import load_config
from src.utils.prompts.ua.tutor_agent import przypadki_template, verb_template


class IntroTool(BaseTool):
    name = "інтро"
    description = "інструмент для початку розмови, вітає користувача, використовуй цей інструмент для початку розмови"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        config = load_config()
        intro = config.intro_template
        return intro

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class SentenceCheckTool(BaseTool):
    name = "інструмент перевірки речення"
    description = "інструмент для перевірки речення, використовуй цей інструмент для перевірки речення"

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
    name = "інструмент для вправ з дієслівами"
    description = (
        "Інструмент для вправ з дієслівами, використовуй цей інструмент для вправ з дієслівами, "
        "Давай практикувати відмінювання дієслів. "
        "Інструмент використовується для правил коньюгації дієслів в польскій мовію"
        "Інструмент використовується для практики коньюгації дієслів в польскій мові"
    )

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

        v_template = PromptTemplate(input_variables=["history", "input"], template=verb_template)

        llm = ChatOpenAI(model_name="gpt-4", temperature=0.1, streaming=True)
        qa_chain = ConversationChain(
            llm=llm,
            verbose=True,
            memory=memory,
            # combine_docs_chain_kwargs={"prompt": qa_template},
        )
        qa_chain.prompt = v_template

        return qa_chain.run(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class PrzypadkiPractiseTool(BaseTool):
    name = "відмінювання прикметників і іменників"
    description = (
        "Інструмент для вправ відмінювання прикметників і іменників, використовуй цей інструмент "
        "для вправ відмінювання прикметників і іменників, "
        "Давай практикувати відмінювання прикметників і іменників. "
        "Інструмент використовується для правил відмінювання прикметників і іменників в польскій мовію"
        "Інструмент використовується для практики відмінювання прикметників і іменників в польскій мові"
    )

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

        v_template = PromptTemplate(input_variables=["history", "input"], template=przypadki_template)

        llm = ChatOpenAI(model_name="gpt-4", temperature=0.1, streaming=True)
        qa_chain = ConversationChain(
            llm=llm,
            verbose=True,
            memory=memory,
            # combine_docs_chain_kwargs={"prompt": qa_template},
        )
        qa_chain.prompt = v_template

        return qa_chain.run(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
