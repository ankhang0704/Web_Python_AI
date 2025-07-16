from django.urls import path
from Web_AI import views


urlpatterns = [
    path("", views.home, name="home"),
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
    path("index/", views.index, name='index'),
    path('process_message/', views.process_chat_message, name='process_chat_message'),
    path("predict_location/", views.predict_location, name="predict_location"),
    path("signup/", views.signup, name='signup'),
    path("chat_history/", views.chat_history, name='chat_history'),
    path('history/<int:session_id>/', views.session_detail, name='session_detail'),
]
