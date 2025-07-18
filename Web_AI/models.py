from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

# Create your models here.
class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    start_time = models.DateTimeField(default=timezone.now)

    title = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"Phiên chat của {self.user.username} lúc {self.start_time.strftime('%Y-%m-%d %H:%M')}"

class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, related_name='contents', on_delete=models.CASCADE)
    
    user_message = models.TextField()
    ai_response = models.TextField(blank=True, null=True)
    image_caption = models.TextField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    

    def __str__(self):
        return f"Nội dung chat lúc {self.timestamp.strftime('%Y-%m-%d %H:%M')} - Phiên: {self.session.id}"
