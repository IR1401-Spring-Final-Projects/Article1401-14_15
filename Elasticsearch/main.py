from elasticsearch import Elasticsearch
import json
import os
import gc

# client = Elasticsearch(hosts="http://127.0.0.1:9200")
#
# print(client.indices.exists(index="papers-index"))
#
# print(client.info())
#
# with open("clean_data.json", "r") as f:
#     json_file = json.load(f)
#
# for i in json_file:
#     doc = {
#         'paperId': json_file[i]["paperId"],
#         'title': json_file[i]["title"].lower(),
#         'abstract': json_file[i]["abstract"].lower(),
#         'year': json_file[i]["year"],
#         'authors': json_file[i]["authors"],
#         'fieldsOfStudy': json_file[i]["fieldsOfStudy"].lower(),
#         'referenceCount': json_file[i]["referenceCount"],
#         'citationCount': json_file[i]["citationCount"],
#         'references': json_file[i]["references"]
#     }
#     resp = client.index(index="papers-index", id=i, document=doc)

# print(json_file["9405cc0d6169988371b2755e573cc28650d14dfe"]["paperId"])
# print(json_file["9405cc0d6169988371b2755e573cc28650d14dfe"]["title"])
# print(json_file["9405cc0d6169988371b2755e573cc28650d14dfe"]["abstract"])
# print(json_file["9405cc0d6169988371b2755e573cc28650d14dfe"]["year"])
# print(json_file["9405cc0d6169988371b2755e573cc28650d14dfe"]["authors"])
# print(json_file["9405cc0d6169988371b2755e573cc28650d14dfe"]["referenceCount"])
# print(json_file["9405cc0d6169988371b2755e573cc28650d14dfe"]["citationCount"])

reduced_data = {}
bagheri = ["NLP.json", "natural_language_processing_transformers.json", "Language_Models.json", "transformers.json"]
for f in bagheri:
    data = json.load(open(f, "r"))
    for key in data:
        reduced_data[key] = {"paperId": key,
                             "title": data[key]["title"].lower(),
                             "abstract": data[key]["abstract"].lower() if data[key]["abstract"] else "",
                             "year": data[key]["year"],
                             "authors": [{"authorId": l["authorId"], "name": l["name"]} for l in data[key]["authors"]],
                             "fieldsOfStudy": [i.lower() for i in data[key]["fieldsOfStudy"] if i] if data[key][
                                 "fieldsOfStudy"] else [],
                             "citationCount": data[key]["citationCount"],
                             "referenceCount": data[key]["referenceCount"],
                             "references": [{"paperId": k["paperId"], "title": k["title"].lower()} for k in
                                            data[key]["references"]]}
    del data
    gc.collect()
json.dump(reduced_data, open("./clean_data.json", "w"))
del reduced_data
gc.collect()
