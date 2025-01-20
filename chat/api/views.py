from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.conf import settings
from .llm import Chatting
from .chat import saveTTS

@require_GET
def adchat_message(request):
    message = request.GET.get('message', '')
    
    if not message:
        return JsonResponse({'error': '메시지가 없습니다.'}, status=400)
    
    chat = Chatting()
    
    response = chat.send_message(message)
    try:
        print(f"saveTTS 호출: {response}")
        file_name = saveTTS(response)
        if not file_name:
            return JsonResponse({'error': 'TTS 파일 생성 실패'}, status=500)
        tts_file_url = f"{settings.MEDIA_URL}{file_name}"
    except Exception as e:
        print(f"saveTTS 실행 중 오류 발생: {e}")
        return JsonResponse({'error': 'TTS 생성 중 오류 발생'}, status=500)

    return JsonResponse({'response': response, 'tts_url': tts_file_url})


@require_GET
def chchat_message(request):
    message = request.GET.get('message', '')
    
    if not message:
        return JsonResponse({'error': '메시지가 없습니다.'}, status=400)
    
    chat = Chatting()
    
    response = chat.send_message(message, False, True)
    try:
        print(f"saveTTS 호출: {response}")
        file_name = saveTTS(response)
        if not file_name:
            return JsonResponse({'error': 'TTS 파일 생성 실패'}, status=500)
        tts_file_url = f"{settings.MEDIA_URL}{file_name}"
    except Exception as e:
        print(f"saveTTS 실행 중 오류 발생: {e}")
        return JsonResponse({'error': 'TTS 생성 중 오류 발생'}, status=500)

    return JsonResponse({'response': response, 'tts_url': tts_file_url})