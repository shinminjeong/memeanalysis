from django.conf.urls import include, url
from django.contrib import admin
from . import views

urlpatterns = [
    #url(r'^$', views.main, name='main'),
    url(r'^$', views.cluster, name='cluster'),
    url(r'^media/$', views.media, name='media'),
    # url(r'^download/$', views.download, name='download'),
    # url(r'^cluster/$', views.cluster, name='cluster'),
    # url(r'^cluster/more/$', views.cluster_more, name='cluster_more'),
    # url(r'^cluster/json/', views.cluster_json, name='cluster_json'),
    # url(r'^cluster/info/$', views.info, name='info'),
]
