from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.conf import settings
from .llm import Chatting
from .chat import saveTTS
import hashlib
import os

chat = Chatting()

@require_GET
def adchat_message(request):
    message = request.GET.get('message', '')
    
    if not message:
        return JsonResponse({'error': '메시지가 없습니다.'}, status=400)
    
    response = chat.send_message(message)
    return JsonResponse({'response': response})


@require_GET
def chchat_message(request):
    message = request.GET.get('message', '')
    
    if not message:
        return JsonResponse({'error': '메시지가 없습니다.'}, status=400)
    
    response = chat.send_message(message, False, True)
    return JsonResponse({'response': response})

@require_GET
def mktts(request):
    """
    메시지 기반 TTS 음성 파일을 생성하거나 기존 파일 반환.
    """
    message = request.GET.get('message', '').strip()
    if not message:
        return JsonResponse({'error': 'TTS 생성 메시지가 없습니다.'}, status=400)
    
    try:
        hash_object = hashlib.md5(message.encode('utf-8'))
        file_name = f"{hash_object.hexdigest()}.mp3"
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        if os.path.exists(file_path):
            print(f"기존 TTS 파일 사용: {file_path}")
        else:
            print(f"TTS 생성 시작: {message}")
            saveTTS(message, file_path)
            print(f"TTS 파일 생성 완료: {file_path}")

        tts_file_url = f"{settings.MEDIA_URL}{file_name}"
        return JsonResponse({'tts_url': tts_file_url})
    except Exception as e:
        print(f"TTS 처리 중 오류 발생: {e}")
        return JsonResponse({'error': 'TTS 생성 중 오류 발생'}, status=500)