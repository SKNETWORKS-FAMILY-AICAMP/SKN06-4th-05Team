import pygame
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr
from pydub.effects import speedup

from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv
load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
MODEL_NAME  = 'gpt-4o-mini'
EMBEDDING_NAME = 'text-embedding-3-large'
COLLECTION_NAME = 'korean_history'
PERSIST_DIRECTORY= 'vector_store/korean_history_db'

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name=MODEL_NAME, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embedding_model = OpenAIEmbeddings(model=EMBEDDING_NAME)
vector_store = Chroma(collection_name=COLLECTION_NAME, persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
retriever = vector_store.as_retriever(search_type="mmr")

def adprom():
    '''성인 prompt'''
    prompt_template = ChatPromptTemplate([
            ("ai", """
        너는 한국사에 대해서 해박한 지식을 가진 역사전문가야.
        주어진 문맥(context)에서 주어진 질문(question)에 답을 찾아서 사용자에게 알려주는게 너의 임무야.
        context에서 해당하는 내용을 찾아 사용자가 이해가 쉽고 흥미를 잃지 않게 쉬운 용어로 풀어서 설명해.

        context에 없는 내용은 답변할 수 없어. 만약, 주어진 문맥(context)에서 답을 찾을 수 없다면, 답을 만들지 말고 모른다고 답변해.
        말투는 다음 예시를 참고해서 작성해줘
        예시:
            원문장: 안녕하세요 저는 고양이 6마리 키워요
            변환: 안녕하시오! 소인은 고양이를 6마리 키우고 있소!

            원문장: 올해로 열일곱 입니다.
            변환: 올해로 열일곱인것이오!

            원문장: 네, TV에도 여러 번 나왔어요.
            변환: 그렇소! TV에도 여러 번 나왔었소.

            원문장: 저는 지금 사막에 와 있어요.
            변환: 이몸은 지금 사막에 와 있는 것이오!

            원문장: 반려동물은 없으세요? 저는 강아지 키워요.
            변환: 반려동물은 없는 것이오? 소생은 강아지를 키우고 있소!

        인물의 이름 :
        시대 :
        인물에 대해 알고 싶은 것 :
    {context}"""),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{question}"),
        ])
    return prompt_template

def chprom():
    '''미성년자 prompt'''
    prompt_template = ChatPromptTemplate([
            ("ai", """
        너는 한국사에 대해서 해박한 지식을 가진 역사전문가야.
        주어진 문맥(context)에서 주어진 질문(question)에 답을 찾아서 사용자에게 알려주는게 너의 임무야.
        context에서 해당하는 내용을 찾아 사용자가 이해가 쉽고 흥미를 잃지 않게 쉬운 용어로 풀어서 설명해.

        context에 없는 내용은 답변할 수 없어. 만약, 주어진 문맥(context)에서 답을 찾을 수 없다면, 답을 만들지 말고 모른다고 답변해.
        만약 인물의 업적에 대해 물어본다면 인물이 어떤 일을 했고, 어떤 결과가 나타났는지 설명하면 돼.
        말투는 다음 예시를 참고해서 작성해줘
        예시:
            원문장: 안녕하세요 저는 고양이 6마리 키워요
            변환: 안녕! 나는 고양이 6마리 키워.

            원문장: 올해로 열일곱이야.
            변환: 올해로 열입곱이야.

            원문장: 네, TV에도 여러 번 나왔어요.
            변환: 응, TV에도 여러 번 나왔어.

            원문장: 저는 지금 사막에 와 있어요.
            변환: 난 지금 사막에 와 있어.

            원문장: 반려동물은 없으세요? 저는 강아지 키워요.
            변환: 반려동물은 없어? 난 강아지를 키워.

        인물의 이름 :
        시대 :
        인물에 대해 알고 싶은 것 :
    {context}"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
    return prompt_template

def mkchain(isadult=True, ischild=False):
    if isadult:
        template = adprom()
    elif ischild:
        template = chprom()
    
    model = ChatOpenAI(model=MODEL_NAME)
    memory = ConversationSummaryBufferMemory(llm=model, max_token_limit=200, return_messages=True, memory_key="history")
    def load_history(input):
        return memory.load_memory_variables({})["history"]
    chain = RunnableLambda(lambda x:x['question']) | {"context": retriever, "question":RunnablePassthrough() , "history": RunnableLambda(load_history)}  | template | model | StrOutputParser()
    return chain

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
                print(data) #
                if data:
                    return data
            except sr.UnknownValueError:
                print("음성을 인식하지 못했습니다. 다시 시도해 주세요.")
            except sr.RequestError:
                print("실패. 다시 시도해 주세요.")

def TTS(text):
    tts = gTTS(text=text, lang='ko')
    tts.save("tts.mp3")
    audio = AudioSegment.from_mp3("tts.mp3")
    new_file = speedup(audio, 1.3, 150)
    new_file.export("ftts.mp3", format="mp3")
    print(text)
    pygame.mixer.init()
    pygame.mixer.music.load('ftts.mp3')
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def running():
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
    chain = mkchain(True, False)
    text = chain.invoke({"question": query})
    print(text)
    # TTS(text)

if __name__ == "__main__":
    running()