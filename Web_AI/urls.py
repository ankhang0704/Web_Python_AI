from django.urls import path
from Web_AI import views


urlpatterns = [
    path("", views.home, name="home"),
    path("Web_AI/<name>", views.hello_there, name="hello_there"),
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
    path("index/", views.index, name='index'),
]
