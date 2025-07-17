# import re
# import os
# from django.utils.timezone import datetime
# from django.http import HttpResponse
# from django.shortcuts import render
# from keras.models import load_model
# import numpy as np
import os
import json
import logging
import urllib.parse
logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.http import HttpResponse
from django.utils.timezone import datetime
from django.http import JsonResponse, Http404
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib import messages


from .ai_images import generate_caption, encode_image
from .ai_service import predict
from .models import ChatSession, ChatMessage



@login_required
def index(request):
    # Xóa session_id cũ khi người dùng vào lại trang chính để bắt đầu cuộc trò chuyện mới
    if 'chat_session_id' in request.session:
        del request.session['chat_session_id']
    return render(request, 'Web_AI/index.html')

@login_required
@require_http_methods(["POST"])
def predict_location(request):
    """
    Xử lý dự đoán địa điểm từ hình ảnh
    """
    try:
        # Kiểm tra xem có file hình ảnh không
        if 'image' not in request.FILES:
            return JsonResponse({
                'error': 'Không có file hình ảnh được gửi lên.'
            }, status=400)
        
        uploaded_file = request.FILES['image']
        description = request.POST.get('description', '')
        
        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
        if uploaded_file.content_type not in allowed_types:
            return JsonResponse({
                'error': 'Chỉ chấp nhận file hình ảnh (JPEG, PNG, WEBP).'
            }, status=400)
        
        # Validate file size (10MB max)
        if uploaded_file.size > 10 * 1024 * 1024:
            return JsonResponse({
                'error': 'File quá lớn. Vui lòng chọn file nhỏ hơn 10MB.'
            }, status=400)
        
        # Tạo đường dẫn lưu file
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'chat_uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Tạo tên file unique
        import uuid
        file_extension = os.path.splitext(uploaded_file.name)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        save_path = os.path.join(upload_dir, unique_filename)
        
        # Lưu file
        with open(save_path, 'wb+') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        try:
            # Encode image và gọi AI
            photo = encode_image(save_path)
            
            # Tạo prompt cho AI nếu có description
            if description:
                prompt = f"Dự đoán địa điểm trong hình ảnh này. Mô tả thêm: {description}"
            else:
                prompt = "Dự đoán địa điểm trong hình ảnh này."
            
            # Gọi hàm AI để dự đoán
            prediction = generate_caption(photo)


            
            # Log để debug
            logger.info(f"Image uploaded: {unique_filename}")
            logger.info(f"Prediction: {prediction}")
            
            # 1. Tạo một PHIÊN CHAT MỚI cho lần upload ảnh này
            new_session = ChatSession.objects.create(
                user=request.user,
                title=f"Hỏi đáp về ảnh: {prediction}" # Đặt tiêu đề cho phiên chat
            )
            
            # 2. Lưu ID của phiên chat này vào session của Django
            #    để các tin nhắn chat sau biết phải lưu vào đâu.
            request.session['chat_session_id'] = new_session.id
            
            # 3. Tạo tin nhắn đầu tiên cho phiên chat này
            ChatMessage.objects.create(
                session=new_session,
                user_message="Đã tải lên một hình ảnh để nhận diện.", # Tin nhắn của người dùng
                image_caption=prediction,    
                ai_response=f"Tôi nhận diện được đây là: {prediction}. Bạn muốn hỏi gì thêm?", # Câu trả lời của bot
            )

            return JsonResponse({
                'prediction': prediction,
                'type': 'image_prediction',
                'session_id': new_session.id
            })
            
        except Exception as e:
            logger.error(f"Error in AI prediction: {str(e)}")
            return JsonResponse({
                'error': 'Lỗi khi xử lý hình ảnh. Vui lòng thử lại.'
            }, status=500)
            
        finally:
            # Xóa file tạm nếu cần (tùy chọn)
            try:
                if os.path.exists(save_path):
                    # Có thể giữ lại file hoặc xóa sau một thời gian
                    pass
            except:
                pass
    
    except Exception as e:
        logger.error(f"Error in predict_location: {str(e)}")
        return JsonResponse({
            'error': 'Lỗi server. Vui lòng thử lại.'
        }, status=500)

def home(request):
    return render(request, "Web_AI/home.html")
  
def about(request):
    return render(request, "Web_AI/about.html")

def contact(request):
    return render(request, "Web_AI/contact.html")

@login_required
def process_chat_message(request):
    """
    View này xử lý các tin nhắn chat được gửi bằng AJAX/Fetch.
    """
    if request.method == 'POST':
        try:
            # Lấy dữ liệu JSON được gửi từ JavaScript
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()

            if not user_message:
                return JsonResponse({'error': 'Tin nhắn không được để trống.'}, status=400)
            # 1. Lấy ID của phiên chat hiện tại từ session của Django
            session_id = request.session.get('chat_session_id')
        
            if not session_id:
                # Nếu không có session, tạo một session mới cho cuộc trò chuyện không có ảnh
                chat_session = ChatSession.objects.create(
                    user=request.user,
                    title=f"Hỏi đáp: {user_message[:30]}..." # Lấy 30 ký tự đầu làm tiêu đề
                )
                request.session['chat_session_id'] = chat_session.id
            else:
                try:
                    # Lấy phiên chat đã tồn tại từ DB
                    chat_session = ChatSession.objects.get(id=session_id, user=request.user)
                except ChatSession.DoesNotExist:
                    return JsonResponse({'error': 'Phiên làm việc không hợp lệ. Vui lòng làm mới trang.'}, status=400)
            
            # Gọi hàm AI để lấy câu trả lời
            bot_response = predict(user_message)


            # 2. Lưu cặp hỏi-đáp này vào phiên chat đã xác định
            ChatMessage.objects.create(
                session=chat_session,
                user_message=user_message,
                ai_response=bot_response
            )
            # Trả về câu trả lời dưới dạng JSON
            return JsonResponse({'response': bot_response})
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Dữ liệu gửi lên không hợp lệ.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Lỗi server: {e}'}, status=500)

    # Nếu không phải POST request
    return JsonResponse({'error': 'Yêu cầu không hợp lệ.'}, status=405)

#login 
def signup(request):
    """
    View xử lý việc đăng ký người dùng mới.
    """
    if request.method == 'POST':
        # Nếu người dùng gửi dữ liệu lên
        form = UserCreationForm(request.POST)
        if form.is_valid():
            # Nếu form hợp lệ, lưu người dùng vào database
            user = form.save()
            # Tự động đăng nhập cho người dùng ngay sau khi đăng ký thành công
            login(request, user)
            # Chuyển hướng về trang chủ
            return redirect('home') # Thay 'chat_interface' bằng name của trang chủ của bạn
    else:
        # Nếu là GET request, chỉ hiển thị form trống
        form = UserCreationForm()
        
    # Truyền form ra template
    return render(request, 'registration/signup.html', {'form': form})

#trang xem lai lich sử chat
@login_required
def chat_history(request):
    """
    Hiển thị lịch sử chat của người dùng.
    """
    # Lấy tất cả các phiên chat của người dùng hiện tại
    chat_sessions = ChatSession.objects.filter(user=request.user).order_by('-start_time')
    
    # Truyền dữ liệu vào template
    return render(request, 'Web_AI/chat_history.html', {'chat_sessions': chat_sessions})

@login_required
def chat_session_detail(request, session_id):
    """
    Hiển thị chi tiết một phiên chat cụ thể.
    """
    try:
        # Lấy phiên chat theo ID
        chat_session = ChatSession.objects.get(id=session_id, user=request.user)
        
        # Lấy tất cả các tin nhắn trong phiên chat này
        chat_messages = ChatMessage.objects.filter(session=chat_session).order_by('created_at')
        
        # Truyền dữ liệu vào template
        return render(request, 'Web_AI/chat_session_detail.html', {
            'chat_session': chat_session,
            'chat_messages': chat_messages
        })
    
    except ChatSession.DoesNotExist:
        return HttpResponse("Phiên chat không tồn tại.", status=404)
    
# # trong views.py
# @login_required
# def session_detail(request, session_id):
#     try:
#         # Đảm bảo người dùng chỉ xem được session của chính họ
#         session = ChatSession.objects.get(id=session_id, user=request.user)
#         # Lấy tất cả tin nhắn trong session đó, sắp xếp theo thời gian
#         messages = session.contents.all().order_by('timestamp')
#         return render(request, 'Web_AI/session_detail.html', {'session': session, 'messages': messages})
#     except ChatSession.DoesNotExist:
#         from django.http import Http404
#         raise Http404("Không tìm thấy phiên chat.")
    
@login_required
def session_detail(request, session_id):
    """
    Hiển thị nội dung chi tiết của một phiên chat cụ thể.
    """
    try:
        # Lấy phiên chat, đảm bảo nó thuộc về người dùng đang đăng nhập để bảo mật
        # get_object_or_404 là một shortcut tiện lợi của Django
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        
        # Lấy tất cả các tin nhắn trong session đó, sắp xếp theo thời gian tăng dần
        # Nhớ rằng 'contents' là related_name bạn đã đặt trong model ChatMessage
        messages = session.contents.all().order_by('timestamp')
        
        # Truyền dữ liệu ra template
        return render(request, 'Web_AI/session_detail.html', {
            'session': session,
            'messages': messages
        })
    except Http404:
        # Nếu không tìm thấy session hoặc session không thuộc về user,
        # Django sẽ tự động ném ra lỗi 404 (trang không tồn tại).
        # Bạn cũng có thể chuyển hướng về trang lịch sử nếu muốn.
        return redirect('chat_history')

@login_required
@require_http_methods(["POST"]) # Chỉ cho phép xóa bằng phương thức POST để bảo mật
def delete_session(request, session_id):
    """
    Xóa một phiên chat cụ thể.
    """
    try:
        # Lấy đúng phiên chat của người dùng đang đăng nhập
        session_to_delete = get_object_or_404(ChatSession, id=session_id, user=request.user)
        session_to_delete.delete()
        # Gửi một thông báo thành công về cho template
        messages.success(request, 'Đã xóa cuộc hội thoại thành công!')
    except Http404:
        messages.error(request, 'Không tìm thấy cuộc hội thoại hoặc bạn không có quyền xóa.')
    except Exception as e:
        messages.error(request, f'Đã có lỗi xảy ra: {e}')

    # Sau khi xóa, chuyển hướng người dùng về lại trang lịch sử
    return redirect('chat_history')

# ==========================================================
#   VIEW MỚI: XÓA TOÀN BỘ LỊCH SỬ
# ==========================================================
@login_required
@require_http_methods(["POST"])
def delete_all_history(request):
    """
    Xóa toàn bộ lịch sử chat của người dùng hiện tại.
    """
    try:
        # Tìm và xóa tất cả các phiên chat của người dùng
        sessions = ChatSession.objects.filter(user=request.user)
        count = sessions.count()
        sessions.delete()
        if count > 0:
            messages.success(request, f'Đã xóa thành công toàn bộ {count} cuộc hội thoại!')
        else:
            messages.info(request, 'Không có lịch sử nào để xóa.')
    except Exception as e:
        messages.error(request, f'Đã có lỗi xảy ra khi xóa lịch sử: {e}')

    return redirect('chat_history')