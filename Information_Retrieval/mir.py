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
class IR:
    def __init__(self,source_path):
        self.source_path
    
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
    
        
    


