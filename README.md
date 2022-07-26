# Article1401-14_15

## Elasticsearch
As you know we use Elasticsearch in this project. Elasticsearch is the distributed, RESTful search and analytics engine. We use it for retrieve data's by the search utilities thatj it gave us. The version of elasticsearch we use in our project is V7.3.2. Also our project connect to elasticsearch via a Python library that also called elasticsearch too, this python library is V3.10.

Before using elasticsearch we set the mapping of Article documents, such as paperId,title, abstract, and ETC. One of the most important features of Elasticsearch is that it tries to get out of way and let we start exploring your data as quickly as possible. For using that features we have to set some settings of our index of documents. Some of the setting that we set in our index are described below.

### Analyzer
ELasticsearch give us some features to analyse the data to having better search. Notice that the settings we set of an index are directly related to search, if the setting such as tokenizer, analyzer and ETC set correctly the result of search would have the maximum score. The analyzer that we use are Ngram and Tokenizer.

#### Tokenizer
Tokenizer let us to set some filters in our data before they index in elasticsearch. They are some filters that we use: EnglishStopWord, word_delimiter, and ngram.

#### Ngram
We use some Ngram features that elastic let us to use, min_gram and max_gram. Also we set max_diff_ngram to having better search.

### Search data using elasticsearch
In this project we used the search tool that elastic developed. Match query is one the search queries that elastic use to find data. In match query we set a field that we want to search on it and the title that we want to found. Elastic use this search query on our index and show us the resuly ny their score. The document with maximum score show up first. The score that each document had, is related to the mapping that we set at first to create an index.

Here you can see the pyhton file of elastic client that we used in our project.
[Code](https://github.com/IR1401-Spring-Final-Projects/Article1401-14_15/blob/main/Elasticsearch/main.py)
