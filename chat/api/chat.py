import os
import time
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr
from pydub.effects import speedup
from django.conf import settings

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
            
            원문장 : 오늘 날씨 정말 좋지 않아요?
            변환 : 오늘 날씨가 참 좋지 않소?

            원문장 : 비가 올 것 같은데 우산 가져왔어요?
            변환 : 비가 올 것 같은데, 우산을 가져왔소?

            원문장 : 점심에 뭐 먹을 거예요?
            변환 : 점심에 무엇을 먹을 것이오?

            원문장 : 요즘 어떤 음식에 빠져 있나요?
            변환 : 요즘 어떤 음식에 빠져 있소?

            원문장 : 주말에 뭐 하면서 보내요?
            변환 : 주말에는 무엇을 하면서 보내오?

            원문장 : 최근에 재미있는 책을 읽었어요.
            변환 : 최근에 재미있는 책을 읽었소.

            원문장 : 요즘 인기 있는 영화 봤어요?
            변환 : 요즘 인기 있는 영화를 보았소?

            원문장 : 추천할만한 영화가 있나요?
            변환 : 추천할 만한 영화가 있소?

            원문장 : 매일 운동해요?
            변환 : 매일 운동을 하오?

            원문장 : 헬스장에 다니고 있어요?
            변환 : 헬스장에 다니고 있소?

            원문장 : 다음 여행 계획이 있나요?
            변환 : 다음 여행 계획이 있소?

            원문장 : 최근에 어디 다녀왔어요?
            변환 : 최근에 어디를 다녀왔소?

            원문장 : 일은 잘 돼가요?
            변환 : 일은 잘 되고 있소?

            원문장 : 오늘 학교에서 어떤 일이 있었어요?
            변환 : 오늘 학교에서 어떤 일이 있었소?

            원문장 : 최근에 산 것 중 가장 마음에 드는 게 뭐예요?
            변환 : 최근에 산 것 중에서 가장 마음에 드는 것이 무엇이오?

            원문장 : 어디서 옷을 주로 사요?
            변환 : 어디서 옷을 주로 사시오

            원문장 : 커피는 주로 어떻게 마셔요?
            변환 : 커피는 어떻게 마시오?

            원문장 : 새로운 카페를 찾았어요.
            변환 : 새로운 카페를 찾았소.

            원문장 : 요즘 어떤 음악 들어요?
            변환 : 요즘 어떤 음악을 듣고 있소?

            원문장 : 좋아하는 밴드나 가수가 있어요?
            변환 : 좋아하는 밴드나 가수가 있소?

            원문장 : 요즘 건강은 어때요?
            변환 : 요즘 건강은 어떠하오?

            원문장 : 충분히 쉬고 있어요?
            변환 : 충분히 쉬고 있소?

            원문장 : 새로 나온 스마트폰 봤어요?
            변환 : 새로 나온 스마트폰을 보았소?

            원문장 : 요즘 어떤 앱을 많이 사용해요?
            변환 : 요즘 어떤 앱을 많이 사용하고 있소?

            원문장 : 최근 뉴스에서 본 게 있나요?
            변환 : 최근 뉴스에서 본 것이 있소?

            원문장 : 요즘 핫한 주제가 뭐예요?
            변환 : 요즘 핫한 주제가 무엇이오?

            원문장 : 벌써 가을이네요. 시간이 정말 빠르죠?
            변환 : 벌써 가을이 왔소. 시간이 참 빠르지 않소?

            원문장 : 계절마다 어떤 활동을 좋아해요?
            변환 : 계절마다 어떤 활동을 좋아하오?

            원문장 : 오늘 입은 옷 멋져요!
            변환 : 오늘 입은 옷이 멋지오!

            원문장 : 좋아하는 스타일이 있나요?
            변환 : 좋아하는 스타일이 있소?


            원문장 : 새로운 사람을 만나는 걸 좋아해요?
            변환 : 새로운 사람을 만나는 것을 좋아하오?

            원문장 : 모임에 자주 나가나요?
            변환 : 모임에 자주 나가시오?

            원문장 : 가족들과 자주 연락해요?
            변환 : 가족들과 자주 연락하시오?

            원문장 : 이번 주말에 가족과 함께할 계획이 있어요?
            변환 : 이번 주말에 가족과 함께할 계획이 있소?

            원문장 : 취업 준비는 잘 돼가요?
            변환 : 취업 준비는 잘 되어 가고 있소?

            원문장 : 면접에서 어떤 질문을 받았어요?
            변환 : 면접에서 어떤 질문을 받았소?

            원문장 : 오늘 수업에서 뭘 배웠어요?
            변환 : 오늘 수업에서 무엇을 배웠소?

            원문장 : 과제는 다 했나요?
            변환 : 과제는 다 했소?

            원문장 : 요즘 어떤 게임을 즐겨요?
            변환 : 요즘 어떤 게임을 즐기고 있소?

            원문장 : 같이 게임해볼래요?
            변환 : 같이 게임을 해 볼 생각이 있소?

            원문장 : 반려동물 키워요?
            변환 : 반려동물을 키우고 있소?

            원문장 :동물원에 가본 지 오래됐네요.
            변환 : 동물원에 가본 지 오래되었구려.

            원문장 : 어떤 기술을 배우고 싶어요?
            변환 : 어떤 기술을 배우고 싶소?

            원문장 : 새로운 언어를 배우고 있어요.
            변환 : 새로운 언어를 배우고 있소.

            원문장 : 명절 계획은 뭐예요?
            변환 : 명절 계획은 무엇이오?

            원문장 : 명절 음식 중에 뭐가 제일 좋아요?
            변환 : 명절 음식 중에 무엇이 제일 좋소?

            원문장 : 스트레스를 어떻게 풀어요?
            변환 : 스트레스를 어떻게 풀고 있소?

            원문장 : 요즘 스트레스 받을 일이 많아요.
            변환 : 요즘 스트레스를 받을 일이 많소.

            원문장 : 보통 몇 시에 자요?
            변환 : 보통 몇 시에 잠자리에 드시오?

            원문장 : 잠을 푹 못 잤어요.
            변환 : 잠을 푹 못 잤구려.

            원문장 : 주말에 모임이 있어요.
            변환 : 주말에 모임이 있소.

            원문장 : 친구들과 언제 만날 계획이에요?
            변환 : 친구들과 언제 만날 계획이오?

            원문장 : 친구들하고 어떻게 지내요?
            변환 : 친구들과 어떻게 지내고 있소?

            원문장 : 오랜만에 친구를 만났어요.
            변환 : 오랜만에 친구를 만났소.

            원문장 : 출퇴근할 때 교통이 어때요?
            변환 : 출퇴근할 때 교통이 어떻소?

            원문장 : 대중교통 이용해요?
            변환 : 대중교통을 이용하고 있소?

            원문장 : 하루를 어떻게 계획하나요?
            변환 : 하루를 어떻게 계획하고 있소?

            원문장 : 시간을 잘 관리하는 편이에요?
            변환 : 시간을 잘 관리하는 편이오?

            원문장 : 어릴 때 어떤 꿈이 있었어요?
            변환 : 어릴 때 어떤 꿈을 가지고 있었소?

            원문장 : 지금도 꿈이 바뀌지 않았어요?
            변환 : 지금도 꿈이 바뀌지 않았소?

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

def saveTTS(text, file_path):
    try:
        tts = gTTS(text=text, lang='ko')
        tts.save(file_path)
        audio = AudioSegment.from_mp3(file_path)
        new_file = speedup(audio, 1.3, 150)
        new_file.export(file_path, format="mp3")
        print(f"최종 TTS 파일 저장 완료: {file_path}")
    except Exception as e:
        print(f"TTS 생성 중 오류 발생: {e}")
        raise e

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