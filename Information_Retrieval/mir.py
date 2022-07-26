import json
import pandas as pd
import numpy as np
import tensorflow
import enum
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
from typing import List,Tuple
import pickle
import spacy
from collections import defaultdict
from gensim.models import KeyedVectors
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
import pandas as pd
import sklearn
import os
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import Counter
from collections import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from scipy import sparse
import networkx as nx
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, TextClassificationPipeline

# nltk.download()
# python3 -m spacy download en_core_web_sm
# pip install -U sentence-transformers

source_path = "./"
def address_resolver(add):
    return source_path + add

def flatten(l : List[List]) -> List:
    return [item for sublist in l for item in sublist]

def is_int(s):
    try:
        int(s)
        return True
    except:
        return False

class Query_type(enum.Enum):
    AUTHOR = "Author Based"
    TITLE = "Title Based"
    ABSTRACT = "Abstract Based"
class Boolean_IR:
    def __init__(self,docs):
        print("boolean search loading modules")
        self.author_to_id = json.load(open("../Information_Retrieval/DATA/P3/author_to_id.json","r"))
        self.author_to_doc = json.load(open("../Information_Retrieval/DATA/P3/author_to_doc.json","r"))
        self.documents = docs
        self.lemma_title = json.load(open("../Information_Retrieval/DATA/P3/title_lemma.json","r"))
        self.bool_dic_title = json.load(open("../Information_Retrieval/DATA/P3/bool_dic_title.json","r"))
        self.nlp = spacy.load("en_core_web_sm")
        self.title_tokenizer = lambda s : [token.lemma_ for token in self.nlp(s) if token.lemma_ not in self.nlp.Defaults.stop_words ]

    def word_tokenize_author(self,t : str) -> List:
        res = word_tokenize(t)
        if (res[-1] != "."):
            return res
        res[-2] = res[-2]+res[-1]
        return res[:-1]

    def pre_process_authors(self) -> None:
        print("boolean search loading authors preprocess")
        self.all_names = list(set(flatten([self.word_tokenize_author(key) for key in self.author_to_id if not is_int(key)])))
        i = iter(range(1,len(self.all_names)+1))
        self.w_mapping = defaultdict(lambda : next(i))
        self.bool_dic_author = defaultdict(lambda : [])
        list(map(lambda x : self.w_mapping[x],self.all_names))
        removed_key = []
        for key in self.author_to_id:
            if not is_int(key) and is_int(self.author_to_id[key]) and key:
                i = self.author_to_id[key]
                self.bool_dic_author[i] = np.array([self.w_mapping[w] for w in self.word_tokenize_author(key)])
            else:
                removed_key.append(key)
        for x in removed_key:
            del self.author_to_id[x]
    def pre_process_title(self) -> None:
        print("boolean search loading title preprocess")
        for key in self.bool_dic_title:
            self.bool_dic_title[key] = np.array(self.bool_dic_title[key])

    def title_ir(self,wk:str , k):
        words = np.array([self.lemma_title.get(w,0) for w in wk])
        titles = [(key,np.sum([np.sum([item == self.bool_dic_title[key] for item in words ])])) for key in self.documents if type(self.documents[key]["title"]) == str]
        return sorted(titles , key = lambda x : x[1],reverse=True)[k[0]:k[1]]

    def author_ir(self,input_wk:str,k) -> List:
        names_map = np.array([self.w_mapping.get(w,0) for w in input_wk])
        authors = [(key,np.sum([np.sum([name == self.bool_dic_author[self.author_to_id[key]] for name in names_map ])])) for key in self.author_to_id]
        return sorted(authors , key = lambda x : x[1],reverse=True)[k[0]:k[1]]

    def query(self,type : Query_type , input_string:str , k) -> Tuple[List,List]:
        input_string = input_string.lower()
        if type == Query_type.TITLE:
            mapping = self.title_ir(self.title_tokenizer(input_string.strip().lower()), k)[k[0]:k[1]]
            return mapping
        elif type == Query_type.AUTHOR:
            names =  self.author_ir(self.word_tokenize_author(input_string.strip()),k)
            articles = flatten([[self.documents[id]["paperId"] for id in self.author_to_doc[self.author_to_id[name[0]]]] for name in names])[k[0]:k[1]]
            return (articles,names)





class TF_IDF_IR:
    def __init__(self,docs):

        self.documents = docs
        print("loading tf-idf model modules")
        self.lemma_title = json.load(open("../Information_Retrieval/DATA/P3/title_lemma.json","r"))
        self.lemma_abs = json.load(open("../Information_Retrieval/DATA/P3/abstract_lemma.json","r"))
        self.idf_abs = json.load(open("../Information_Retrieval/DATA/P3/idf_abstract.json","r"))
        self.idf_title = json.load(open("../Information_Retrieval/DATA/P3/idf_title.json","r"))
        self.tf_title = json.load(open("../Information_Retrieval/DATA/P3/title_tf.json","r"))
        self.tf_abs = json.load(open("../Information_Retrieval/DATA/P3/asb_tf.json","r"))
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = lambda s : [token.lemma_ for token in self.nlp(s) if token.lemma_ not in self.nlp.Defaults.stop_words ]
        for key in self.tf_title:
            self.tf_title[key] = {int(k) : float(self.tf_title[key][k]) for k in self.tf_title[key]}
        for key in self.tf_abs:
            self.tf_abs[key] = {int(k) : float(self.tf_abs[key][k]) for k in self.tf_abs[key]}
        self.lemma_title = {key : int(self.lemma_title[key]) for key in self.lemma_title}
        self.lemma_abs = {key : int(self.lemma_abs[key]) for key in self.lemma_abs}
        self.idf_abs =  {int(key) : float(self.idf_abs[key]) for key in self.idf_abs}
        self.idf_title =  {int(key) : float(self.idf_title[key]) for key in self.idf_title}

    def process_q(self,q : List , tf , idf , k) -> List[Tuple]:
        without_expansion = sorted([(key,sum([tf[key].get(wq,0) * idf.get(wq,0) for wq in q])) for key in tf], key = lambda x : x[1] , reverse=True)[k[0]:k[1]]
        return without_expansion


    def query(self,type : Query_type , input_string:str , k) -> List:
        wk = self.tokenizer(input_string.strip().lower())
        if type == Query_type.TITLE:
            q = [int(self.lemma_title.get(w,0)) for w in wk]
            result = self.process_q(q,self.tf_title,self.idf_title , k)
        elif type == Query_type.ABSTRACT:
            q = [int(self.lemma_abs.get(w,0)) for w in wk]
            result = self.process_q(q,self.tf_abs,self.idf_abs , k)
        return result





def softmax(x):
    y = np.exp(x - np.max(x))
    return y / y.sum()

class Fast_text_TF_IDF_IR:
    def __init__(self,docs,t = "lemma" , c_soft = True):
        self.documents = docs
        self.t = t
        self.mapping = None
        self.idf = None
        self.train_data_path = None
        print("loading fasttext requirments")
        if t == "lemma":
            self.nlp = spacy.load("en_core_web_sm")
            self.tokenizer = lambda s : [token.lemma_ for token in self.nlp(s)]
            self.train_data_path = "../Information_Retrieval/fasttext/fasttext_data.txt"
            self.mapping = json.load(open("../Information_Retrieval/DATA/P3/abstract_lemma.json","r"))
            self.idf = json.load(open("../Information_Retrieval/DATA/P3/idf_abstract.json","r"))
        else:
            self.tokenizer = lambda s : [token for token in word_tokenize(s)]
            self.train_data_path = "../Information_Retrieval/fasttext/fasttext_not_lemma_data.txt"
            self.mapping = json.load(open("../Information_Retrieval/DATA/P3/abstract_not_lemma.json","r"))
            self.idf = json.load(open("../Information_Retrieval/DATA/P3/idf_abstract_not_lemma.json","r"))
        self.emmbeding = None
        self.mapping = {key : int(self.mapping[key]) for key in self.mapping}
        self.idf = {int(key) : float(self.idf[key]) for key in self.idf}
        self.c_soft = lambda x : x
        if c_soft:
            self.c_soft = lambda x : softmax(x)
        self.doc_emb = {}
        self.dim = 300
    def is_c_(self,w):
        try:
            self.emmbeding[w]
            return True
        except:
            return False


    def preprocess(self,pre = False , ws = 5 ,epoch = 20 ,lr = 0.1, dim = 200):
        self.dim = dim
        if not pre:
            print("training fasttext module")
            os.system("rm ../Information_Retrieval/fasttext/word_embedding.*")
            print("\n making fastext")
            os.chdir('../Information_Retrieval/fasttext/fastText')
            os.system("make")
            os.chdir('../..')
            print(os.getcwd())
            print("./fasttext/fastText/fasttext module")
            os.system(f"../Information_Retrieval/fasttext/fastText/fasttext skipgram -dim {dim} -ws {ws} -epoch {epoch} -lr {lr} -input {self.train_data_path} -output ../Information_Retrieval/fasttext/word_embedding")
            os.system("rm ../Information_Retrieval/fasttext/word_embedding.bin")
            self.emmbeding = KeyedVectors.load_word2vec_format("../Information_Retrieval/fasttext/word_embedding.vec")
            for key in self.documents:
                article = self.documents[key]
                abstract = article["abstract"]
                if (type(abstract) == str):
                    try:
                        word = self.tokenizer(abstract)
                        matrix = np.array([self.emmbeding[w] for w in word if self.is_c_(w)]).reshape(-1,self.dim)
                        c = np.array([self.idf.get(self.mapping.get(w,0),0) for w in word if self.is_c_(w)]).reshape(1,-1)
                        c = self.c_soft(c)
                        self.doc_emb[key] = np.matmul(c,matrix).tolist()[0]
                    except:
                        print(key,c)
            open("../Information_Retrieval/fasttext/doc_embedding.json","w").write(json.dumps(self.doc_emb))

        self.emmbeding = KeyedVectors.load_word2vec_format("../Information_Retrieval/fasttext/word_embedding.vec")
        self.doc_emb = json.load(open("../Information_Retrieval/fasttext/doc_embedding.json","r"))
        self.doc_emb = {key : np.array(self.doc_emb[key]).reshape(1,self.dim) for key in self.doc_emb}

    def process_q(self,q : np.array , expansion = False) -> List[Tuple]:
        without_expansion = sorted([(key,np.abs(distance.cosine(q,self.doc_emb[key]))) for key in self.doc_emb],key = lambda x : x[1])
        if not expansion :
            return without_expansion
        near = without_expansion[:10]
        far = without_expansion[-10:]
        q = 0.6 * q + 0.5 * np.mean([self.doc_emb[a[0]]for a in near],axis = 0) - 0.1 * np.mean([self.doc_emb[a[0]]for a in far], axis = 0)
        return sorted([(key,np.abs(distance.cosine(q,self.doc_emb[key]))) for key in self.doc_emb],key = lambda x : x[1])

    def query(self, input_string:str , k, expansion = False) -> List:
        word = self.tokenizer(input_string.strip().lower())
        matrix = np.array([self.emmbeding[w] for w in word if self.is_c_(w)]).reshape(-1,self.dim)
        c = self.c_soft(np.array([self.idf.get(self.mapping.get(w,0),0) for w in word if self.is_c_(w)]).reshape(1,-1))
        q = np.matmul(c,matrix)[0]
        article_id = self.process_q(q , expansion)[k[0]:k[1]]
        return article_id

source = "./"
f_source = lambda s : source+"/"+s
class Transformer:
  def __init__(self,docs,model_name = '../Information_Retrieval/DATA/sentence-transformers_all-MiniLM-L12-v2/'):
    print(f"Transformer\ndownloading model {model_name}")
    self.model = SentenceTransformer(model_name)
    self.documents = docs
    self.representation = None

  def preprocess(self,pre_use = False):
    if not pre_use:
      docs = []
      keys = []
      print(f"creating representation for docs")
      for key in self.documents:
        abstract = self.documents[key]["abstract"]
        if type(abstract) == str:
          docs.append(abstract)
          keys.append(key)
      embeddings = self.model.encode(docs)
      self.representation = {}
      for key, embedding in zip(keys, embeddings):
        self.representation[key] = embedding.tolist()
      addr = f_source("../Information_Retrieval/DATA/P3/transformer.json")
      print(f"saving docs_rep in {addr}")
      open(addr,"w").write(json.dumps(self.representation))
    print(f"loading docs_rep")
    self.representation = json.load(open(f_source("../Information_Retrieval/DATA/P3/transformer.json"),"r"))
    self.representation = {key : np.array(self.representation[key]) for key in self.representation }
  def query(self,input_str:str , k , expansion = False):
    q = self.model.encode(input_str)
    without_expansion = sorted([(key,np.abs(distance.cosine(q,self.representation[key]))) for key in self.representation],key = lambda x : x[1])
    if not expansion:
        article_id = without_expansion[k[0]:k[1]]
        article =  [self.documents[id[0]] for id in article_id]
        return (article_id,article)
    near = without_expansion[:10]
    far = without_expansion[-10:]
    q = 0.6 * q + 0.5 * np.mean([self.representation[a[0]]for a in near],axis = 0) - 0.1 * np.mean([self.representation[a[0]]for a in far],axis = 0)
    return sorted([(key,np.abs(distance.cosine(q,self.representation[key]))) for key in self.representation],key = lambda x : x[1])[k[0]:k[1]]


class Page_Ranking_Hits:
    def __init__(self):
        objective = "article"
        self.ref_matrix = None
        self.articles = sparse.load_npz("../Information_Retrieval/DATA/P5/articles_sparse.npz")
        self.objective = self.articles if objective == "article" else self.authors
        representation = json.load(open(f_source("../Information_Retrieval/DATA/P5/article_mapping.json"),"r"))
        self.mapping = {int(representation[doc]):doc for doc in representation}

    def compute_page_rank(self, alpha = 0.9):
        graph = nx.from_numpy_array(A=self.objective.toarray(), create_using=nx.DiGraph)
        self.pr = nx.pagerank(G=graph, alpha=alpha)

    def compute_hits(self):
        graph = nx.from_numpy_array(A=self.objective.toarray(), create_using=nx.DiGraph)
        self.hub, self.authority = nx.hits(G=graph)

    def tops_pages(self, k = 10):
        res = sorted(self.pr.items(),key = lambda x : x[1] , reverse = True)[:k]
        return [self.mapping[int(ar[0])] for ar in res]

    def cal_cites(self):
        return np.asarray(np.sum(self.objective,axis = 0)).reshape(-1)

    def top_hubs(self, k = 10):
        return sorted(self.hub.items(),key = lambda x : x[1] , reverse = True)[:k]

    def top_auth(self, k = 10):
        return sorted(self.authority.items(),key = lambda x : x[1] , reverse = True)[:k]

    def cal_ref(self):
        return np.asarray(np.sum(self.objective,axis = 1)).reshape(-1)

MAIN_DATA_PATH = "../DATA/clean_data.json"
CLUSTER_DATA_PATH = "../DATA/clustring_data.csv"
CLASSIFICATION_DATA_PATH = "../DATA/classification_data.csv"


class IR:
    def __init__(self):
        print("loading requirments ... ")
        print("loading main data ... ")

        self.main_data = json.load(open(address_resolver(MAIN_DATA_PATH),"r"))

        print("loading clustring data ... ")
        self.clustring_data = pd.read_csv(address_resolver(CLUSTER_DATA_PATH))
        self.cluster_labels_map = {0:"cs.LG" , 1:"cs.CV" , 2:"cs.AI" , 3:"cs.RO" , 4:"cs.CL"}
        self.kmeas_map_label = {0: 2, 1: 2, 2: 1, 3: 3, 4: 2, 5: 1, 6: 1, 7: 4, 8: 1, 9: 1, 10: 1, 11: 1}
        self.cluster_model = pickle.load(open("../Information_Retrieval/DATA/P4/finalized_cluster_model.sav", 'rb'))
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
        self.transformer = Transformer(self.main_data,'../Information_Retrieval/DATA/sentence-transformers_all-MiniLM-L12-v2/')
        self.transformer.preprocess(pre_use = True)
        self.bert_model = self.transformer.model

        print("page_ranking_algorithm loading")
        self.page_hits_articles = Page_Ranking_Hits()
        self.page_hits_articles.compute_page_rank(0.9)
        self.page_hits_articles.compute_hits()

        print("loading classification data ... ")
        self.classification_model_name  = 'distilbert-base-uncased'
        self.classification_model = AutoModelForSequenceClassification.from_pretrained("../Information_Retrieval/DATA/P4/classification_model", from_tf=True)
        self.classification_classes = {
                        'LABEL_0' : 'cs.CV',
                        'LABEL_1' : 'cs.LG',
                        'LABEL_2' : 'stat.ML'
                    }
        self.classification_class_categories = {v: k for k, v in self.classification_classes.items()}
        self.classification_tokenizer = AutoTokenizer.from_pretrained(self.classification_model_name)
        self.classification_pipeline = TextClassificationPipeline(model=self.classification_model, tokenizer=self.classification_tokenizer, return_all_scores=False)

        print("Finished loading packages.")






    def classification(self,text):
        prediction = self.classification_pipeline(text)[0]
        predicted_class = self.classification_classes[prediction['label']]
        return predicted_class



    def clustring(self,text):
        # abstract_bert = bert_model.encode(data_abstract,device = "cuda")
        # titles_bert = bert_model.encode(data_title,device = "cuda")
        #concated_data_bert = np.array([np.array([abstract_bert[i],titles_bert[i]]).reshape(-1) for i in range((abstract_bert.shape[0]))])
        # text
        # class
        abstract_bert = self.bert_model.encode([text])
        titles_bert = self.bert_model.encode(["title"])
        concated_data_bert = np.array([np.array([abstract_bert[i],titles_bert[i]]).reshape(-1) for i in range((abstract_bert.shape[0]))])
        return self.cluster_labels_map[self.kmeas_map_label[self.cluster_model.predict(concated_data_bert)[0]]]


    def search(self,text,type_text,query_expansion = False,mode = "bert" , range_q = (0,40)):# abstract title author
        #mode = bert , tf-idf , fasttext , boolean
        #text_type = abstract , title , author
        # range_q = (start , end)
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

    def best_articles(self,k = 10):# page_rank,hits
        return self.page_hits_articles.tops_pages(k = 10)
        #self.page_hits_articles.s
        # type_query = page_rank , hits ,
        # k = numbers
        # mode = artcile , author
