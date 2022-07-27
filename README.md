# Article1401-14_15

## requirments
for runnig this project your system should have ability to run `make` command line. then download sentence-transformers_all-MiniLM-L12-v2 and classification model from [here](https://drive.google.com/drive/folders/1hS7yPD2SJGtwAs3G-nnezMRogxRgux2W)

requirments for project are included in requirments.txt in conda form and pip form.

## Search 
As we suppposed to we have 4 types of searching besides elasticsearch we merged all of the modules and provided nice api functions for backend so everything is nice and clean.


`note that if you want to clone our project from github you should first download sentence_transformers and classsification_model and set it in Information_Retrieval/DATA/P4/classification_model, and Information_Retrieval/DATA/sentence-transformers and you should also copy huggingface to ~/.cache`

we first load and init all of previous modules in code bellow.



```py
class IR:
    def __init__(self):
        print("loading requirments ... ")
        print("loading main data ... ")

        self.main_data = json.load(open(address_resolver(MAIN_DATA_PATH),"r"))

        print("loading clustring data ... ")
        self.clustring_data = pd.read_csv(address_resolver(CLUSTER_DATA_PATH))
        self.cluster_labels_map = {0:"cs.LG" , 1:"cs.CV" , 2:"cs.AI" , 3:"cs.RO" , 4:"cs.CL"}
        self.kmeas_map_label = {0: 2, 1: 2, 2: 1, 3: 3, 4: 2, 5: 1, 6: 1, 7: 4, 8: 1, 9: 1, 10: 1, 11: 1}
        self.cluster_model = pickle.load(open("DATA/P4/finalized_cluster_model.sav", 'rb'))
        print("cluster_model = ",self.cluster_model)

        print("loading Boolean search model")
        self.boolean_ir = Boolean_IR(self.main_data)
        self.boolean_ir.pre_process_authors()
        self.boolean_ir.pre_process_title()

        print("loading tf-idf search model")
        self.tf_idf_raw = TF_IDF_IR(self.main_data)

        print("loading fasttext module")
        self.fast_text = Fast_text_TF_IDF_IR(self.main_data,t = "lemma")
        print("process fasttext module")
        self.fast_text.preprocess(pre = True ,dim=400, epoch=20 , lr = 0.06 , ws = 10 )

        print("Transformers loading")
        self.transformer = Transformer(self.main_data,'./DATA/sentence-transformers_all-MiniLM-L12-v2/')
        self.transformer.preprocess(pre_use = True)
        self.bert_model = self.transformer.model

        print("page_ranking_algorithm loading")
        self.page_hits_articles = Page_Ranking_Hits()
        self.page_hits_articles.compute_page_rank(0.9)
        self.page_hits_articles.compute_hits()

        print("loading classification data ... ")
        self.classification_model_name  = 'distilbert-base-uncased'
        self.classification_model = AutoModelForSequenceClassification.from_pretrained("DATA/P4/classification_model", from_tf=True)
        self.classification_classes = {
                        'LABEL_0' : 'cs.CV',
                        'LABEL_1' : 'cs.LG',
                        'LABEL_2' : 'stat.ML'
                    }
        self.classification_class_categories = {v: k for k, v in self.classification_classes.items()}
        self.classification_tokenizer = AutoTokenizer.from_pretrained(self.classification_model_name)
        self.classification_pipeline = TextClassificationPipeline(model=self.classification_model, tokenizer=self.classification_tokenizer, return_all_scores=False)

        print("Finished loading packages.")s
```

And here are functions you can call for searching for data.
 for seeing what each search does you should read previous docs and report for P3,4,5

```py
    def classification(self,text):
        prediction = self.classification_pipeline(text)[0]
        predicted_class = self.classification_classes[prediction['label']]
        return predicted_class



    def clustring(self,text):
        abstract_bert = self.bert_model.encode([text])
        titles_bert = self.bert_model.encode(["title"])
        concated_data_bert = np.array([np.array([abstract_bert[i],titles_bert[i]]).reshape(-1) for i in range((abstract_bert.shape[0]))])
        return self.cluster_labels_map[self.kmeas_map_label[self.cluster_model.predict(concated_data_bert)[0]]]


    def search(self,text,type_text,query_expansion = False,mode = "bert" , range_q = (0,40)):
        if mode == "bert":
            return self.transformer.query(text,k = range_q , expansion=query_expansion)
        elif mode == "fasttext":
            return self.fast_text.query(text,range_q,query_expansion)
        elif mode == "tf_idf":
            qt = Query_type.ABSTRACT
            if type_text == "title":
                qt = Query_type.TITLE
            return self.tf_idf_raw.query(qt, text , range_q)
        elif mode == "boolean":
            qt = Query_type.TITLE
            if type_text == "author":
                qt = Query_type.AUTHOR
            return self.boolean_ir.query(qt,text,range_q)

    def best_articles(self,k = 10):
        return self.page_hits_articles.tops_pages(k = 10)
```

## Clustring
for clustering we enhanced our previous model and our purity reached 70 from 68.

for enhancement I used LDA (Linear Discriminative Analysis) for processing and then used Kmeans on that.

## Classification
for P4 we developed 2 methods for this purpose; 1. Logistic Regression, 2. transformers (distilbert-case-uncased).
for method 1 we were using tf-idf vectors which need all of the train data to build a vector for the test; so we should wait for several minutes to classify only 1 document and so on. 
thus here we use the transformer model that was trained in P4; keep in mind that we increased the f1-score of the model by training it for 5 epochs.

## Page_Ranking 
nothing changed so just review previous report on P5

## Query_Expansion

we coordinated with our assigned TA mr. Balapour that this type of query is meaningful only for fasttext and transformer model and we apply this for our project only on fasttext and transformer.

here is the code:

```py
near = without_expansion[:10]
        far = without_expansion[-10:]
        q = 0.6 * q + 0.5 * np.mean([self.representation[a[0]]for a in near],axis = 0) - 0.1 * np.mean([self.representation[a[0]]for a in far],axis = 0)
        article_id = sorted([(key,np.abs(distance.cosine(q,self.representation[key]))) for key in self.representation],key = lambda x : x[1])[k[0]:k[1]]
        return [i[0] for i in article_id]```
```




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

