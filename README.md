# Article1401-14_15

## Elasticsearch
As you know, we use Elasticsearch in this project. Elasticsearch is the distributed, RESTful search and analytics engine. We use it to retrieve data through the search utilities it gave us. The version of elasticsearch we use in our project is V7.3.2. Also, our project connects to elasticsearch via a Python library called elasticsearch; this python library is V3.10.

Before using elasticsearch, we set the mapping of Article documents, such as paperId, title, abstract, ETC. One of the most important features of Elasticsearch is that it tries to get out of the way and lets us start exploring your data as quickly as possible. To use that features, we have to set some settings of our index of documents. Some of the settings that we put in our index are described below.

### Analyzer
ELasticsearch gives us some features to analyze the data for a better search. Notice that the settings we set for an index are directly related to the search. If the setting such as tokenizer, analyzer, ETC are specified correctly, the result of the search would have the maximum score. The analyzer that we use is Ngram and Tokenizer.

#### Tokenizer
Tokenizer lets us set some filters in our data before they index in elasticsearch. We use some filters: EnglishStopWord, word_delimiter, and ngram.

#### Ngram
We use some Ngram features that elastic let us use, min_gram and max_gram. Also, we set max_diff_ngram to have a better search.

### Search data using elasticsearch
In this project, we used the search tool that elastic developed. Match query is one of the search queries that elastic use to find data. In the match query, we set a field on which we want to search and the title we want to see. Elastic use this search query on our index and shows us the results by their score. The document with the maximum score shows up first. Each record's score is related to the mapping we set at first to create an index.

Here you can see the python file of the elastic client we used in our project.
[Code](https://github.com/IR1401-Spring-Final-Projects/Article1401-14_15/blob/main/Elasticsearch/main.py)

