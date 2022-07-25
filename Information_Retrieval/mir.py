import json
import pandas as pd
import numpy as np
import enum
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
from typing import List,Tuple
import spacy
from collections import defaultdict
from gensim.models import KeyedVectors
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
import pandas as pd
import sklearn
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
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

# nltk.download()
# python3 -m spacy download en_core_web_sm

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
        self.author_to_id = json.load(open("DATA/P3/author_to_id.json","r"))
        self.author_to_doc = json.load(open("DATA/P3/author_to_doc.json","r"))
        self.documents = docs
        self.lemma_title = json.load(open("DATA/P3/title_lemma.json","r"))
        self.bool_dic_title = json.load(open("DATA/P3/bool_dic_title.json","r"))
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
            mapping = self.title_ir(self.title_tokenizer(input_string.strip().lower()), k)
            articles = [self.documents[id[0]] for id in mapping]
            return (articles,mapping)
        elif type == Query_type.AUTHOR:
            names =  self.author_ir(self.word_tokenize_authoe(input_string.strip()),k) 
            articles = flatten([[self.documents[id] for id in self.author_to_doc[self.author_to_id[name[0]]]] for name in names])
            return (articles[k[0]:k[1]],names)





class TF_IDF_IR:
    def __init__(self,docs):

        self.documents = docs
        print("loading tf-idf model modules")
        self.lemma_title = json.load(open("DATA/P3/title_lemma.json","r"))
        self.lemma_abs = json.load(open("DATA/P3/abstract_lemma.json","r"))
        self.idf_abs = json.load(open("DATA/P3/idf_abstract.json","r"))
        self.idf_title = json.load(open("DATA/P3/idf_title.json","r"))
        self.tf_title = json.load(open("DATA/P3/title_tf.json","r"))
        self.tf_abs = json.load(open("DATA/P3/asb_tf.json","r"))
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
        return sorted([(key,sum([tf[key].get(wq,0) * idf.get(wq,0) for wq in q])) for key in tf], key = lambda x : x[1] , reverse=True)[:k]
        

    def query(self,type : Query_type , input_string:str , k : int = 10) -> List:
        wk = self.tokenizer(input_string.strip().lower())
        if type == Query_type.TITLE:
            q = [int(self.lemma_title.get(w,0)) for w in wk]
            result = self.process_q(q,self.tf_title,self.idf_title , k)
        elif type == Query_type.ABSTRACT:
            q = [int(self.lemma_abs.get(w,0)) for w in wk]
            result = self.process_q(q,self.tf_abs,self.idf_abs , k)
        articles = [self.documents[id[0]] for id in result]
        return (articles,result)





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
            self.train_data_path = "./fasttext/fasttext_data.txt"
            self.mapping = json.load(open("DATA/P3/abstract_lemma.json","r"))
            self.idf = json.load(open("DATA/P3/idf_abstract.json","r"))
        else:
            self.tokenizer = lambda s : [token for token in word_tokenize(s)]
            self.train_data_path = "./fasttext/fasttext_not_lemma_data.txt"
            self.mapping = json.load(open("DATA/P3/abstract_not_lemma.json","r"))
            self.idf = json.load(open("DATA/P3/idf_abstract_not_lemma.json","r"))
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
            os.system("rm ./fasttext/word_embedding.*")
            os.system(f"./fasttext/fastText/fasttext skipgram -dim {dim} -ws {ws} -epoch {epoch} -lr {lr} -input {self.train_data_path} -output ./fasttext/word_embedding")
            os.system("rm ./fasttext/word_embedding.bin")
            self.emmbeding = KeyedVectors.load_word2vec_format("./fasttext/word_embedding.vec")
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
            open("./fasttext/doc_embedding.json","w").write(json.dumps(self.doc_emb))

        self.emmbeding = KeyedVectors.load_word2vec_format("./fasttext/word_embedding.vec")
        self.doc_emb = json.load(open("./fasttext/doc_embedding.json","r"))
        self.doc_emb = {key : np.array(self.doc_emb[key]).reshape(1,self.dim) for key in self.doc_emb}

    def process_q(self,q : np.array) -> List[Tuple]:
        return sorted([(key,np.abs(distance.cosine(q,self.doc_emb[key]))) for key in self.doc_emb],key = lambda x : x[1])
        

    def query(self, input_string:str , k : int = 10) -> List:
        word = self.tokenizer(input_string.strip().lower())
        matrix = np.array([self.emmbeding[w] for w in word if self.is_c_(w)]).reshape(-1,self.dim)
        c = self.c_soft(np.array([self.idf.get(self.mapping.get(w,0),0) for w in word if self.is_c_(w)]).reshape(1,-1))
        q = np.matmul(c,matrix)[0]
        article_id = self.process_q(q)[:k]
        articles = [self.documents[id[0]] for id in article_id]
        return (article_id,articles)

        
MAIN_DATA_PATH = "../DATA/clean_data.json"
CLUSTER_DATA_PATH = "../DATA/clustring_data.csv"

class IR:
    def __init__(self):
        print("loading requirments ... ")
        print("loading main data ... ")
        self.main_data = json.load(open(address_resolver(MAIN_DATA_PATH),"r"))
        print("loading clustring data ... ")
        self.clustring_data = pd.read_csv(address_resolver(CLUSTER_DATA_PATH))
        print("loading Boolean search model")
        self.boolean_ir = Boolean_IR(self.main_data)
        self.boolean_ir.pre_process_authors()
        self.boolean_ir.pre_process_title()
        print("loading tf-idf search model")
        self.tf_idf_raw = TF_IDF_IR(self.main_data)
        print("loading fasttext module")
        self.fast_text = Fast_text_TF_IDF_IR(self.main_data,t = "lemma")
        print("process fasttext module")
        self.fast_text.preprocess(pre = False ,dim=400, epoch=20 , lr = 0.06 , ws = 10 )


        
    
    def classification(self,text):
        #text class
        pass
    
    def clustring(self,text):#GMM
        # text
        # class
        
        pass
    
    def search(self,text,type_text,query_expansion = False,mode = "bert" , range_q = (0,10)):# abstract title author
        #mode = bert , tf-idf , fasttext , boolean
        #text_type = abstract , title , author
        # range_q = (start , end)
        pass
    
    def best_articles(self,type_query = "page_rank",mode = "article",k = 10):# page_rank,hits
        # type_query = page_rank , hits ,
        # k = numbers
        # mode = artcile , author

        pass



# ir = IR()


