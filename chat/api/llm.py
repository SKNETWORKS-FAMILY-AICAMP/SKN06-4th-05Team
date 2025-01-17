from . import chat

class Chatting:
    """
    대화형 AI 채팅 클래스.
    
    GPT 모델을 사용하여 사용자와 대화를 수행하고, 대화 기록을 관리한다.
    """

    def __init__(self):
        self.adchain = chat.mkchain()
        self.chchain = chat.mkchain(False, True)

    def send_message(self, query:str, isad=True, isch=False):
        """
        사용자 메시지를 처리하고 AI 응답을 반환.
        Parameter:
            message: str 사용자가 입력한 메시지
            history: list - 사용자와 AI간의 이전까지의 대화 기록

        Returns:
            str: AI의 응답 메시지
        """
        text = self.chchain.invoke({"question": query})
        return text