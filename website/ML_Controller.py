import pandas as pd
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.naive_bayes import MultinomialNB

import warnings
warnings.filterwarnings(action='ignore')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from nltk.wsd import lesk
import pickle

#Read dataset
data =pd.read_csv('spam_ham_dataset.csv')


#===========required function for text preprocessing============#
punctuation = u",.?!()-_\"\'\\\n\r\t;:+*<>@#§^$%&|/{}"
stop_words_eng = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tag_dict = {"J": wn.ADJ,
            "N": wn.NOUN,
            "V": wn.VERB,
            "R": wn.ADV}

def filter_tag_by_dic(tag):
    return tag_dict.get(tag[0].upper(), None)

def lemmatize_tagged_words(tag):
    tags= filter_tag_by_dic(tag[1])    
    return lemmatizer.lemmatize(tag[0], tags) if tags is not None else tag[0]

def bag_of_words(sentence, stop_words=None):
    if stop_words is None:
        stop_words = stop_words_eng
    original_words = word_tokenize(sentence)
    tagged_words = nltk.pos_tag(original_words,tagset='universal') 
    original_words = None
    lemmatized_words = [ lemmatize_tagged_words(ow) for ow in tagged_words ]
    tagged_words = None
    cleaned_words = [ word for word in lemmatized_words if (word not in punctuation) and (word not in stop_words) ]
    lemmatized_words = None
    return " ".join(cleaned_words)

#process of all text that is in dataset
data['cleaned_data']=data['text'].apply(lambda string: bag_of_words(string.lower()))

#============Completed Text Preprocessing================================

#============Tokenizer and Vectorizer====================================
#Convert a collection of text documents to a matrix of token counts


cv=CountVectorizer()
X=cv.fit_transform(data.cleaned_data)

#Get the categories
y=data.label

#required file to calculate for model 
pickle.dump(cv,open("countvectorized-transfrom-updated.pkl","wb"))

#============Completed Tokenizer========================================

#============Split Dataset Into Train and Test==========================

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=0)

#============Completed Splitting========================================

#============Training Data and Creating ML Model========================
models={
    "MultinomialNB": {"model":MultinomialNB(), "perf":0}
}

for name, model in models.items():
    model['model'].fit(X_train,y_train)

#============Completed Training=========================================

#============Testing to Training========================================

models_acc = [2]
models_acc.remove(2)
models_acc

for name, model in models.items():
    models_acc.append([name, model["model"].score(X_test, y_test),model["perf"]])
#============Completed Testing=========================================

#===========Finding Best Hyparameter====================================

parameters = {"alpha": [0.2,1,2,5,10], "fit_prior": [True, False]}

grid = GridSearchCV(MultinomialNB(), param_grid=parameters)
grid.fit(X_train,y_train)

alpha, fit_prior = grid.best_params_['alpha'], grid.best_params_['fit_prior']
model = MultinomialNB(alpha = alpha, fit_prior=fit_prior)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

pickle.dump(model,open("spam-model-mnb-updated.pkl","wb"))

#===========Completed Finding Hyperparameter=============================
