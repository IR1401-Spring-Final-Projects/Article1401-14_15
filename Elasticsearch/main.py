from elasticsearch import Elasticsearch
import json
import os
import gc

# Config elasticsearch CLient
client = Elasticsearch(hosts="http://127.0.0.1:9200")
print(client.info())

# Configure the index and the type
request_body = {

    "settings": {
        "index": {
            "max_ngram_diff": 7
        },
        "analysis": {
            "analyzer": {
                "my_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "word_delimiter", "stop", "ngram"
                    ],
                    "min_gram": 3,
                    "max_gram": 10
                }
            },
            "filter": {
                "english_stop": {
                    "type": "stop",
                    "stopwords": "_english_"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "paperId": {
                "type": "text"
            },
            "title": {
                "type": "text",
                "analyzer": "my_analyzer",
                "search_analyzer": "my_analyzer",
                "search_quote_analyzer": "my_analyzer"
            },
            "abstract": {
                "type": "text",
                "analyzer": "my_analyzer",
                "search_analyzer": "my_analyzer",
                "search_quote_analyzer": "my_analyzer"
            },
            "year": {
                "type": "text"
            },
            "authors": {
                "properties": {
                    "authorId": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "name": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    }
                }
            },
            "fieldsOfStudy": {
                "type": "text"
            },
            "citationCount": {
                "type": "long"
            },
            "referenceCount": {
                "type": "long"
            },
            "references": {
                "properties": {
                    "paperId": {
                        "type": "text"
                    },
                    "title": {
                        "type": "text"
                    }
                }
            }
        }
    }
}
print("creating 'papers-index' index...")
client.indices.create(index='papers-index', body=request_body)
print(client.indices.exists(index="papers-index"))

# Read the file that would be index in elasticsearch
with open("clean_data.json", "r") as f:
    json_file = json.load(f)

# Index the data i elasticsearch
for i in json_file:
    doc = {
        'paperId': json_file[i]["paperId"],
        'title': json_file[i]["title"],
        'abstract': json_file[i]["abstract"],
        'year': json_file[i]["year"],
        'authors': json_file[i]["authors"],
        'fieldsOfStudy': json_file[i]["fieldsOfStudy"],
        'referenceCount': json_file[i]["referenceCount"],
        'citationCount': json_file[i]["citationCount"],
        'references': json_file[i]["references"]
    }
    resp = client.index(index="papers-index", id=i, document=doc)

############################
# Ignore the section below #
############################


# counter = 0
# flag = False
# reduced_data = {}
# bagheri = ["NLP.json", "natural_language_processing_transformers.json", "Language_Models.json", "transformers.json"]
# for f in bagheri:
#     data = json.load(open(f, "r"))
#     for key in data:
#         reduced_data[key] = {"paperId": key,
#                              "title": data[key]["title"].lower(),
#                              "abstract": data[key]["abstract"].lower() if data[key]["abstract"] else "",
#                              "year": data[key]["year"],
#                              "authors": [{"authorId": l["authorId"], "name": l["name"]} for l in data[key]["authors"]],
#                              "fieldsOfStudy": [i.lower() for i in data[key]["fieldsOfStudy"] if i] if data[key][
#                                  "fieldsOfStudy"] else [],
#                              "citationCount": data[key]["citationCount"],
#                              "referenceCount": data[key]["referenceCount"],
#                              "references": [{"paperId": k["paperId"], "title": k["title"].lower()} for k in
#                                             data[key]["references"]]}
#         counter += 1
#         if counter >= 11500:
#             flag = True
#             break
#     del data
#     gc.collect()
#     if flag:
#         break
# json.dump(reduced_data, open("./clean_data.json", "w"))
# del reduced_data
# gc.collect()
