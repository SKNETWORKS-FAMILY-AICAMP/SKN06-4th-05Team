from django.http import JsonResponse

from django.views.decorators.http import require_GET
from .llm import Chatting

@require_GET
def adchat_message(request, message):
    """
    대화를 수행하는 API 엔드포인트.
    path parameter로 사용자의 메시지를 받아서 AI의 응답을 반환한다.
    session을 이용해 대화 기록을 관리한다.
    """
    chat = Chatting()
    
    # llm에게 메세지 전송
    response = chat.send_message(message)

    # JsonResponse(dict): HttpResponse 타입
    return JsonResponse({'response': response})

@require_GET
def chchat_message(request, message):
    """
    대화를 수행하는 API 엔드포인트.
    path parameter로 사용자의 메시지를 받아서 AI의 응답을 반환한다.
    session을 이용해 대화 기록을 관리한다.
    """
    chat = Chatting()
    
    # llm에게 메세지 전송
    response = chat.send_message(message, False, True)

    # JsonResponse(dict): HttpResponse 타입
    return JsonResponse({'response': response})
