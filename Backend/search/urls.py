from django.urls import path

from search.views import search, classify, cluster

urlpatterns = [
    path('sadegh/', search, name='search'),
    path('classify/', classify, name='classify'),
    path('cluster/', cluster, name='cluster'),
]
