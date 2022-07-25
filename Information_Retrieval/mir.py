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
import os
from gensim.models import KeyedVectors
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
import pandas as pd
import sklearn
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

MAIN_DATA_PATH = "../DATA/clean_data.json"
CLUSTER_DATA_PATH = "../DATA/clustring_data.json"

class Query_type(enum.Enum):
    AUTHOR = "Author Based"
    TITLE = "Title Based"
class Boolean_IR:
    def __init__(self,docs):
        self.author_to_id = json.load(open("DATA/Module_data/author_to_id.json","r"))
        self.author_to_doc = json.load(open("DATA/Module_data/author_to_doc.json","r"))
        self.documents = docs
        self.lemma_title = json.load(open("DATA/Module_data/title_lemma.json","r"))
        self.bool_dic_title = json.load(open("DATA/Module_data/bool_dic_title.json","r"))
        self.nlp = spacy.load("en_core_web_sm")
        self.title_tokenizer = lambda s : [token.lemma_ for token in self.nlp(s) if token.lemma_ not in self.nlp.Defaults.stop_words ]

    def word_tokenize_author(self,t : str) -> List:
        res = word_tokenize(t)
        if (res[-1] != "."):
            return res
        res[-2] = res[-2]+res[-1]
        return res[:-1]

    def pre_process_authors(self) -> None:
        self.all_names = list(set(self.flatten([self.word_tokenize_author(key) for key in self.author_to_id if not is_int(key)])))
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
        for key in self.bool_dic_title:
            self.bool_dic_title[key] = np.array(self.bool_dic_title[key])


            
    def title_ir(self,wk:str , k : int = 10):
        words = np.array([self.lemma_title.get(w,0) for w in wk])
        titles = [(key,np.sum([np.sum([item == self.bool_dic_title[key] for item in words ])])) for key in self.documents if type(self.documents[key]["title"]) == str]
        return sorted(titles , key = lambda x : x[1],reverse=True)[:k]


    def author_ir(self,input_wk:str,k) -> List:
        names_map = np.array([self.w_mapping.get(w,0) for w in input_wk])
        authors = [(key,np.sum([np.sum([name == self.bool_dic_author[self.author_to_id[key]] for name in names_map ])])) for key in self.author_to_id]
        return sorted(authors , key = lambda x : x[1],reverse=True)[:k]

    def query(self,type : Query_type , input_string:str , k : int = 10) -> Tuple[List,List]:
        input_string = input_string.lower()
        if type == Query_type.TITLE:
            mapping = self.title_ir(self.title_tokenizer(input_string.strip().lower()), k)
            articles = [self.documents[id[0]] for id in mapping]
            return (articles,mapping)
        elif type == Query_type.AUTHOR:
            names =  self.author_ir(self.word_tokenize_authoe(input_string.strip()),k) 
            articles = self.flatten([[self.documents[id] for id in self.author_to_doc[self.author_to_id[name[0]]]] for name in names])
            return (articles[:k],names)

class IR:
    def __init__(self):
        print("loading requirments ... ")
        print("loading main data ... ")
        self.main_data = json.load(open(address_resolver(MAIN_DATA_PATH),"r"))
        print("loading clustring data ... ")
        self.clustring_data = json.load(open(address_resolver(CLUSTER_DATA_PATH),"r"))
        print("loading Boolean search model")
        self.boolean_ir = Boolean_IR(self.main_data)

        
    
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



    


