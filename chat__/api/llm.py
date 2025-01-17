from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from . import chat

class Chatting:
    """
    대화형 AI 채팅 클래스.
    
    GPT 모델을 사용하여 사용자와 대화를 수행하고, 대화 기록을 관리한다.
    """

    def __init__(self, query):
        self.ad_chain = chat.chain_adult()
        self.ch_chain = chat.chain_child()
        # self.tts = chat.TTS(query)

    def send_message(self, query, isadult:bool=True, ischild:bool=False):
        """
        사용자 메시지를 처리하고 AI 응답을 반환.
        Parameter:
            query: 사용자의 질문
            audlt/chlid: 사용자 나이
        Returns:
            str: AI의 응답 메시지
        """
        if isadult:
            chain = self.ad_chain
        elif ischild:
            chain = self.ch_chain
        responese = chain.invoke(query)
        return responese