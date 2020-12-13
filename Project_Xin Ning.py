#!/usr/bin/env python
# coding: utf-8

# In[79]:


#import packages to get the dataset and EDA
get_ipython().system('pip install wordcloud')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from wordcloud import WordCloud
from wordcloud import STOPWORDS

#Get the dataset
train=pd.read_csv("/Users/xin/Desktop/project/quora-insincere-questions-classification/train.csv")
test=pd.read_csv("/Users/xin/Desktop/project/quora-insincere-questions-classification/test.csv")
sample=pd.read_csv("/Users/xin/Desktop/project/quora-insincere-questions-classification/sample_submission.csv")

#A glance at the basic inforamtion
train.info()
test.info()
sample.info()
train.head()
test.head()
sample.head()
train.columns

#more detailed information
train_sample = train.sample(n=10)
train_sample.question_text.head(n=10)

#Know the question type clearly
insincere=train[train.target==1]
sincere=train[train.target==0]


# In[80]:


train.head()
test.head()
sample.head()
train.columns


# In[81]:


test.head()


# In[82]:


insincere.head()


# In[83]:


#more detailed information
train_sample = train.sample(n=10)
train_sample.question_text.head(n=10)


# In[84]:


#Know the question type clearly
insincere=train[train.target==1]
sincere=train[train.target==0]


# In[85]:


# EDA method1 #number and percentage of target sincere questions and insincere questions
target=train['target'].value_counts()
train_len = train.shape[0]
target0= len(train[train.target == 0])
target1= len(train[train.target == 1])
target0_pct= target0/train_len * 100
target1_pct = target1/train_len * 100
print(target)
print(f'The number of sincere questions is {target0}')
print(f'The number of sincere questions is {target1}')
print (f'The precentage of sincere questions is {target0_pct:.2f}%')
print (f'The precentage of insincere questions is {target1_pct:.2f}%')


# In[86]:


#EDA method2 Histgorams
#The percentage of target sincere questions and insincere questions
ax = sns.countplot(x=train['target'], data=train)


# In[87]:


#EDA method3 the pie chart 
values = [[target0], [target1]]
labels = ["Sincere Questions", "Insincere Questions"]
plt.pie(values, labels=labels,autopct='%1.2f%%')
plt.title("Target Distribution")
plt.show()


# In[88]:


#EDA method4
#The question length and chart
train.question_text.str.split().str.len().describe()
#number of words o the question
train["num_words"] = train["question_text"].apply(lambda x: len(str(x).split()))
test["num_words"] = test["question_text"].apply(lambda x: len(str(x).split()))
#barchart
color = ('blue')
countwords = train["num_words"].value_counts()
plt.figure(figsize=(18,9))
sns.barplot(countwords.index,countwords.values, alpha=0.8, color=color[0])
plt.ylabel('Number of Occurrences', fontsize=15)
plt.xlabel('Number of words in a question', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# In[89]:


#EDA method5 Explore the word length
#I leraned how to count from this website https://stackoverflow.com/questions/61110908/how-can-i-use-lambda-to-count-the-number-of-words-in-a-file
# Average length of the words in the text 
train["mean_word_len"] = train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["mean_word_len"] = test["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
#barchart
color=('blue')
countlen = train["mean_word_len"].value_counts()
plt.figure(figsize=(18,9))
sns.barplot(countlen.index, countlen.values, alpha=0.8, color=color[0])
plt.ylabel('Number of Occurrences', fontsize=15)
plt.xlabel('Number of average word length', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# In[90]:


#EDA method6 fancy worldcloud 
#AI learned how to draw a wordcloud form https://www.datacamp.com/community/tutorials/wordcloud-python
wordcloud = WordCloud(max_font_size=1500, max_words=100, background_color="black").generate(str(train.question_text.values))
plt.figure(figsize = (9,9))
plt.title("Word cloud")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[91]:


#EDA method6 Another way but similar wordcloud
text = " ".join(review for review in train.question_text)
print ("There are {} words in the train dataset of question text.".format(len(text)))
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["affect", "now"])
# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(str(train.question_text.values)) 
# Display the generated image(the matplotlib way)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[92]:


wordcloud = WordCloud().generate(str(insincere.question_text.values))
plt.figure(figsize = (10,10))
plt.title("insincere word cloud")
plt.imshow(wordcloud, interpolation='bilinear')


# In[93]:


wordcloud = WordCloud().generate(str(sincere.question_text.values))
plt.figure(figsize = (10,10))
plt.title("insincere word cloud")
plt.imshow(wordcloud, interpolation='bilinear')


# In[94]:


#Model 1:Logistic Regression Model 
#The idea is from here https://zhuanlan.zhihu.com/p/36120960 
#when I just look though internet and nitice that it is a simple way to solve this problem without using any bmbeddings.
#import the packages
import re
from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#Split the traindataset in to train and testdataset
traindata, testdata = train_test_split(train, test_size=0.7)


# In[96]:


#Some question tests have common words in both sincere and insincere questions. So, I try to clean some common words. 
#I learn from https://machinelearningmastery.com/clean-text-machine-learning-python/ 
#and https://towardsdatascience.com/how-to-efficiently-remove-punctuations-from-a-string-899ad4a059fb
#https://gist.github.com/aaronkub/257a1bd9215da3a7221148600d849450
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(be)|(a)|(an)|(the)|(is)|(are)|(am)|(was)|(were)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|")
def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews
train_CLEAN = preprocess_reviews(traindata['question_text'])
test_CLEAN = preprocess_reviews(testdata['question_text'])


# In[97]:


#Convert the questions to the martix of 0 and 1 for the model
Convert = CountVectorizer(binary=True)
Convert.fit(train)
X_train = Convert.transform(train_CLEAN)
X_test = Convert.transform(test_CLEAN)
#Use model fit to train a model and review the model to see the accuracy
T_traindata = traindata['target']
T_testdata = testdata['target']
#establish the model and train model
model_lg = LogisticRegression()
model_lg.fit(X_train, T_traindata)
#evaulate the model
accuracy_score(T_testdata, model_lg.predict(X_test))


# In[98]:


# use model predit to getd a final result
#I learn how to sace predictions to csv from 
#https://stackoverflow.com/questions/34864695/saving-prediction-results-to-csv
test_clean = preprocess_reviews(test['question_text'])
X_test = Convert.transform(test_clean)
Res = model_lg.predict(X_test)
Results_file = pd.DataFrame({"qid" : test['qid'], "prediction" : Res}).to_csv("Results_file.csv", index = None)


# In[99]:


#model2
#I think that when people say something insincere or sincere, there are some common words have to use
#So it's a good way to make computer recognise the frequent use words and let them learn it's insincere
#The idea is from this
#The difficult situation comes, how to let the machine remember words, 
#I firstly think that it can remember the combination of 26 words, which can be reflected by numbers, but this number array way may have some misunderstandings
#because there are 10 nunmbers and 26 letters, when I try to show letters, I have to choose two numbers for a letter, it's too complex
#Then after two days, I think that how must I represent the words in such a detailed way!
#I can just create a chart that use number to represent words not letters, and I can just choose the most frequent using words!
#I get idea about how to make a vocabulary for deep learning from
#https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
#https://www.educative.io/courses/natural-language-processing-ml/N0Wr9zwpEmv
#https://docs.python.org/3/library/tokenize.html

#import packages
get_ipython().system('pip install tqdm')
import warnings
from tqdm import tqdm
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import RNN, LSTM, Dropout, Flatten, Embedding, SpatialDropout1D, Dense, Dropout, Bidirectional
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,f1_score,plot_confusion_matrix


# In[100]:


#as the dataset is really big, when I train the data, it takes muchmore time, so I just select a mini part of data at this step
#and I plan to combine embedings with the dataset
trainmini = train.loc[:10000, :]
traindatam, testdatam = train_test_split(trainmini, test_size=0.3)
#some detailed information
train_sample = trainmini.sample(n=10)
train_sample.question_text.head(n=10)


# In[101]:


# create a Vocabulary using the question_text
def get_vocab(train, num_words=20000):
    #get the dictionary using the data
    tokenizer = Tokenizer(num_words=num_words)
    texts = train.question_text.tolist()
    tokenizer.fit_on_texts([item.lower() for item in texts])
    return tokenizer
T = get_vocab(trainmini)

#check the results
print(T.texts_to_sequences(["I am the best"]))


# In[102]:


#I learn how to use embeddings form Kaggle notebooks
Emb_program = open('/Users/xin/Desktop/project/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt')
embeddings={}
for line in tqdm(Emb_program):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings[word] = coefs
Emb_program.close()
print('Found %s word vectors.' % len(embeddings))


# In[103]:


#onstruct a matrix in next step, 
#this matrix and the previously created dictionary T.word_index must be one-to-one correspondence, 
#and then use this matrix as the initialization of embedding
vocab_size = len(T.word_index.items())
# create a weight matrix for words in training docs
embedding_matrix = np.random.normal(loc=0, scale=1.0, size=(vocab_size+1, 300))
for word, i in T.word_index.items():
    embedding_vector = embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[104]:


#Use the serialization method sequence to realize the padding of the sentence
MAX_LENGTH = 28
train_X = np.array(pad_sequences(T.texts_to_sequences(traindatam.question_text.tolist()), maxlen=MAX_LENGTH, padding = 'post'))
valid_X = np.array(pad_sequences(T.texts_to_sequences(testdatam.question_text.tolist()), maxlen=MAX_LENGTH, padding = 'post'))
train_y, valid_y = np.array(traindatam.target.values), np.array(testdatam.target.values)


# In[105]:


# define model by using keras
#Ilearned how to use some model from: https://zhuanlan.zhihu.com/p/53051992
#https://blog.csdn.net/dream_catcher_10/article/details/48522339
#https://blog.csdn.net/sinat_22510827/article/details/89526712
#https://keras.io/zh/getting-started/sequential-model-guide/
#https://keras.io/api/layers/regularization_layers/spatial_dropout1d/
model2 = Sequential()
model2.add(Embedding(vocab_size+1, 300, input_length=MAX_LENGTH, weights=[embedding_matrix]))
model2.add(SpatialDropout1D(0.3))
model2.add(LSTM(100))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()


# In[106]:


#Train model and model accuracy
model2.fit(train_X, train_y, epochs=2, verbose=1, batch_size=256)


# In[107]:


#Prediction on test data
y_pred=model2.predict_classes(valid_X)
print(classification_report(valid_y,y_pred))


# In[108]:


#model3 the data that used in model3 is almost the same with model2, so I just use the dataset directly 
#https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/#:~:text=Bidirectional%20LSTMs%20are%20an%20extension,LSTMs%20on%20the%20input%20sequence.
#create model
model3=Sequential()
model3.add(Embedding(vocab_size+1, 300, input_length=MAX_LENGTH, weights=[embedding_matrix]))
model3.add(Bidirectional(LSTM(100)))
model3.add(Dropout(0.3))
model3.add(Dense(1,activation='sigmoid'))
model3.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model3.summary())


# In[109]:


#Train model and model accuracy
model3.fit(train_X, train_y, epochs=2, verbose=1, batch_size=256)


# In[110]:


#Prediction on test data
y_pred=model3.predict_classes(valid_X)
print(classification_report(valid_y,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




