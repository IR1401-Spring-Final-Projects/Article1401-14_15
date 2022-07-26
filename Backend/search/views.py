import os
import sys

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from elasticsearch import Elasticsearch

path = os.getcwd()
path = path[:path.rfind('/') + 1]
path += 'Information_Retrieval/'
sys.path.insert(1, path)
import mir

ir = mir.IR()
client = Elasticsearch(hosts="http://127.0.0.1:9200")


@csrf_exempt
def search(request):
    is_elastic = bool(int(request.GET['elastic']))
    search_type = request.GET['type']
    search_by = request.GET['by']
    expression = request.GET['expression']
    expand = bool(int(request.GET['expand']))

    if is_elastic:
        result = client.search(index="papers-index", query={"match": {search_by: expression}})
        response_data = []
        for doc in result.body['hits']['hits']:
            score = doc['_score']
            paper_id = doc['_source']['paperId']
            authors = doc['_source']['authors']
            title = doc['_source']['title']
            abstract = doc['_source']['abstract']
            year = doc['_source']['year']
            response_data.append({
                'id': paper_id,
                'score': score,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'year': year
            })
    else:
        response_data = ir.search(expression, search_by, expand, search_type, (0, 10))
        print(response_data)

    return JsonResponse(data=response_data, safe=False)


@csrf_exempt
def classify(request):
    text = request.POST.get('text')
    resp = ir.classification(text)
    return JsonResponse(data={'class': resp})


@csrf_exempt
def cluster(request):
    text = request.POST.get('text')
    resp = ir.clustring(text)
    return JsonResponse(data={'cluster': resp})
