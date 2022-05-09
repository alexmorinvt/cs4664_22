import numpy as np
import pandas as pd

import string
import re
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 


# def data_preprocessing(text):
#   #lower case normalizing 
#   text=text.lower()
#   #remove punctuations
#   punctuations = string.punctuation
#   #remove stopwords
#   stop_words = set(stopwords.words("english"))
  

#   words=text.split(" ")
#   words = [WordNetLemmatizer().lemmatize(word, "v") for word in words]
#   words = [w for w in words if w not in stop_words and w not in punctuations]

#   clean = " ".join(words)
#   #remove any non alphanum, digit character
#   clean = re.sub("\W+", " ", clean)
#   #remove numbers
#   clean = re.sub(r'[0-9]+', '', clean)
#   #remove single letter
#   clean = re.sub('\s+\S\s+', '', clean)
  
#   clean = re.sub("  ", " ", clean)
#   return clean

def data_preprocessing(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1\. \2', text)  # before lower case
  #lower case normalizing 
    text=text.lower()
  #remove punctuations
    punctuations = string.punctuation
  #remove stopwords
    stop_words = set(stopwords.words("english"))

  # New stop words list 
    customize_stop_words = ['item','part','1a','ii','iii','iv']

  # Mark them as stop words
    for w in customize_stop_words:
        stop_words.add(w)
  

    words=text.split(" ")
    words = [WordNetLemmatizer().lemmatize(word, "v") for word in words]
    words = [w for w in words if w not in stop_words and w not in punctuations]

    clean = " ".join(words)
  #remove any non alphanum, digit character
    clean = re.sub("\W+", " ", clean)
  #remove numbers
    clean = re.sub(r'[0-9]+', " ", clean)
  
 
  #remove single letter
    clean = re.sub('\s+\S\s+', " ", clean)

  #letter repetition (if more than 2)
    clean = re.sub(r' ([a-z])\1{1,} ', " ", clean)
  
  #clean = re.sub("   ", " ", clean)
  #clean = re.sub("  ", " ", clean)

    words=clean.split(" ")
    words = [WordNetLemmatizer().lemmatize(word, "v") for word in words]
    words = [w for w in words if w not in stop_words and w not in punctuations]
    clean = " ".join(words)
    return clean