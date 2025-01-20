from . import chat
from django.conf import settings
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
MODEL_NAME  = 'gpt-4o-mini'
EMBEDDING_NAME = 'text-embedding-3-large'
COLLECTION_NAME = 'korean_history'
PERSIST_DIRECTORY= 'vector_store/korean_history_db'

model = ChatOpenAI(model='gpt-4o-mini')
memory = ConversationSummaryBufferMemory(llm=model, max_token_limit=200, return_messages=True, memory_key="history")
embedding_model = OpenAIEmbeddings(model=EMBEDDING_NAME)
vector_store = Chroma(collection_name=COLLECTION_NAME, persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
retriever = vector_store.as_retriever(search_type="mmr")

class Chatting:
    """
    대화형 AI 채팅 클래스.
    
    GPT 모델을 사용하여 사용자와 대화를 수행하고, 대화 기록을 관리한다.
    """

    def __init__(self):
        self.adprom = chat.adprom()
        self.chprom = chat.chprom()

    def send_message(self, query:str, isad=True, isch=False):
        """
        사용자 메시지를 처리하고 AI 응답을 반환.
        Parameter:
            message: str 사용자가 입력한 메시지
            history: list - 사용자와 AI간의 이전까지의 대화 기록

        Returns:
            str: AI의 응답 메시지
        """
        if isad:
            template = self.adprom
        elif isch:
            template = self.chprom
        def load_history(input):
            print(memory.load_memory_variables({})["history"])
            return memory.load_memory_variables({})["history"]
        chain = RunnableLambda(lambda x:x['question']) | {"context": retriever, "question":RunnablePassthrough() , "history": RunnableLambda(load_history)}  | template | model
        text = chain.invoke({"question": query})
        memory.save_context(inputs={"human": query}, outputs={"ai":text.content})
        return text.content