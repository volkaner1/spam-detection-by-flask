from sre_constants import SUCCESS
from flask import Blueprint, flash, render_template, request
import pickle


import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

#=======text preprocessing variables and function========
punctuation = u",.?!()-_\"\'\\\n\r\t;:+*<>@#§^$%&|/{}"
stop_words_eng = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tag_dict = {"J": wn.ADJ,
            "N": wn.NOUN,
            "V": wn.VERB,
            "R": wn.ADV}

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
        
def filter_tag_by_dic(tag):
    return tag_dict.get(tag[0].upper(), None)

def lemmatize_tagged_words(tag):
    tags= filter_tag_by_dic(tag[1])    
    return lemmatizer.lemmatize(tag[0], tags) if tags is not None else tag[0]    
#========================================================

#==========creating blueprint from pages=================
pages=Blueprint('pages',__name__)
#========================================================

#==============Backend==================================
@pages.route('/',methods=['POST','GET'])
def home():
    model=pickle.load(open('/home/volkan/Desktop/python fullstack/flaskProject7/website/spam-model-mnb-updated.pkl','rb'))
    cv=pickle.load(open('/home/volkan/Desktop/python fullstack/flaskProject7/website/countvectorized-transfrom-updated.pkl','rb'))

    if request.method=='POST':
        mail=request.form["mail"]
        cleaned_mail=bag_of_words(mail)
        cleaned_token_mail=cv.transform([cleaned_mail])
        pred_email=model.predict(cleaned_token_mail)

        if pred_email[0]=='spam':
            flash('SPAM!!!', category='error')
        else:
            flash('NOT SPAM!!!', category='success')

        return render_template("home.html",detection=pred_email[0])
    else:
        return render_template("home.html")
#============================================================
