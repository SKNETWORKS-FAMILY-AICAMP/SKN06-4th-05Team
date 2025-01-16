import re
import os
from glob import glob

from gtts import gTTS
from playsound import playsound
from pydub import AudioSegment
from pydub.effects import speedup
import speech_recognition as sr

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
MODEL_NAME  = 'gpt-4o'
EMBEDDING_NAME = 'text-embedding-3-large'

COLLECTION_NAME = 'korean_history'
PERSIST_DIRECTORY= 'vector_store/korean_history_db'
category = ["사건", "인물"]
name = ["고대", "고려", "근대", "조선", "현대"]

# Split
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name=MODEL_NAME,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# db 연결
embedding_model = OpenAIEmbeddings(model=EMBEDDING_NAME)
vector_store = Chroma(collection_name=COLLECTION_NAME, persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)

def qna(query):
    retriever = vector_store.as_retriever(search_type="mmr")
    # Prompt Template 생성
    messages = [
            ("ai", """
        너는 한국사에 대해서 해박한 지식을 가진 역사전문가야.
        주어진 문맥(context)에서 주어진 질문(question)에 답을 찾아서 사용자에게 알려주는게 너의 임무야.
        context에서 해당하는 내용을 찾아 사용자가 이해가 쉽고 흥미를 잃지 않게 쉬운 용어로 풀어서 설명해.

        context에 없는 내용은 답변할 수 없어. 만약, 주어진 문맥(context)에서 답을 찾을 수 없다면, 답을 만들지 말고 모른다고 답변해.

        인물의 이름 :
        시대 :
        인물에 대해 알고 싶은 것 :
    {context}"""),
            ("human", "{question}"),
        ]
    prompt_template = ChatPromptTemplate(messages)

    # 모델
    model = ChatOpenAI(model=MODEL_NAME)

    # output parser
    parser = StrOutputParser()

    # Chain 구성 retriever(관련 문서 조회) -> prompt_template(prompt 생성) model(정답) -> output parser
    chain = {"context":retriever, "question":RunnablePassthrough()} | prompt_template | model | parser
    return chain.invoke(query)

def TTS(query):
    text = qna(query)
    tts = gTTS(text=text, lang='ko')
    tts.save("tts.mp3")
    audio = AudioSegment.from_mp3("tts.mp3")
    new_file = speedup(audio, 1.3, 150)
    new_file.export("ftts.mp3", format="mp3")
    print(text)
    playsound('ftts.mp3')

def STT():
    recognizer = sr.Recognizer()
    while True:
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("말씀해 주세요. (그만하기: 종료)") # 
            audio = recognizer.listen(source)
            try:
                data = recognizer.recognize_google(audio, language="ko")
                print(data)
                if data:
                    return data
                # session = session_state["prompt"] = data # 세션에 추가
            except sr.UnknownValueError:
                print("음성을 인식하지 못했습니다. 다시 시도해 주세요.")
            except sr.RequestError:
                print("실패. 다시 시도해 주세요.")

if __name__ == "__main__":
    while True:
        num = int(input("1: 음성 인식 2: 직접 입력 > "))
        if num == 1:
            query = STT()
        elif num == 2:
            query = input("질문을 입력하세요(종료: exit) > ")
        
        if query == "exit" or "종료":
            break
        elif len(query) > 3:
            break

    # query = "도시락 폭탄을 던진 사람은 누구인가요?"    
    TTS(query)