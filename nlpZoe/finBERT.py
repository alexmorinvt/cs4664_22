import numpy as np 
import pandas as pd 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

faang=pd.read_csv('../DATA/FAANG_STOCK_NEWS.csv')
netflix = (faang[faang['ticker']=='NFLX'])
netflix=  pd.DataFrame(columns =['date', 'time', 'headline'], data=netflix)

# the below two functions are adopt from https://www.kaggle.com/code/hathalye7/ff-project-finbert/notebook
import torch.nn.functional as F
def SentimentAnalyzer(doc):
    pt_batch = tokenizer(doc,padding=True,truncation=True,max_length=512,return_tensors="pt")
    outputs = model(**pt_batch)
    pt_predictions = F.softmax(outputs.logits, dim=-1)
    return pt_predictions.detach().cpu().numpy()

def findPercentageBySentences(sentenceList):
    posAvg, negAvg, neuAvg = 0, 0, 0
    sentimentArr = SentimentAnalyzer(sentenceList)[0]
    # sentimentArr = np.mean(sentimentArr, axis=0)
    posAvg=sentimentArr[0]
    negAvg=sentimentArr[1]
    neuAvg=sentimentArr[2]
    return {'pos': posAvg, 'neg': negAvg, 'neu' : neuAvg}

#senData={}
#data=pd.DataFrame()
data=[]
for i, line in tqdm(enumerate(netflix['headline'])):
    list=findPercentageBySentences(line)
    #senData[i]=list
    data.append([netflix.iloc[i]['date'], netflix.iloc[i]['time'], line,list['pos'],list['neg'],list['neu']])
    #data.append(['date': netflix.iloc[i]['date'], 'time': netflix.iloc[i]['time'], 'headline':line,'pos':list['pos'],'neg':list['neg'],'neu':list['neu']},ignore_index=True)

df = pd.DataFrame(columns =['date', 'time','headline', 'positive','negative','neutral'], data=data)
df.to_csv('../DATA/netflix_bert_sen.csv')
#print(df)

#senData[0]
