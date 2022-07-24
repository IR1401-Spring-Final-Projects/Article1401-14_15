from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.utils.timezone import now
from elasticsearch import Elasticsearch

client = Elasticsearch("http://localhost:9200")


def search(request):
    print(request.GET)
    is_elastic = bool(request.GET['elastic'])
    search_type = request.GET['type']
    search_by = request.GET['by']
    expression = request.GET['expression']

    print(f'{is_elastic} {search_type} {search_by} {expression}')

    resp = dict(client.info())
    return JsonResponse(data=resp)
