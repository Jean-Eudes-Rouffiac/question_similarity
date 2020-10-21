import pandas as pd
import json
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from num2words import num2words




def get_FQuAD_data(qa_data):
    questions_list = []
    reponses_list = []
    contexts_list = []
    for instance in qa_data['data']:
        for paragraph in instance['paragraphs']:
            context = paragraph['context']
            qas = paragraph['qas']
            for qas_instance in qas:
                question = qas_instance['question']
                answers = qas_instance['answers']
                for answer in answers:
                    ans = answer['text']
                    questions_list.append(question)
                    reponses_list.append(ans)
                    contexts_list.append(context)
    data = pd.DataFrame(list(zip(contexts_list,questions_list,reponses_list)), columns = ['contexte', 'question', 'reponse'])
    return data

def remove_ponctation(object_column):
    ponctuation_list = ['!','#','$','%','&','(',')','*','+',',','.','/',':',';','<','=','>','?','@','[','\'',']','^','_','{','|','}','~','\\', "'"]
    for ponctuation in ponctuation_list:
        object_column = object_column.apply(lambda x: x.replace(ponctuation, ""))
    return object_column

def remove_commons_words(object_column):
    ponctuation_list = ['']
    for ponctuation in ponctuation_list:
        object_column = object_column.apply(lambda x: x.replace(ponctuation, ""))
    return object_column

def data_processing(data, column):
    stemmer= PorterStemmer()
    with open('data/stopwords.json' ,mode="rt",encoding="utf-8") as file:
        stopword = json.load(file)
    stopword_list = list(stopword['words'])
    data[column+'_tokens'] = data[column].apply(lambda x: str(x))
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: str.lower(x))
    data[column+'_tokens'] = remove_ponctation(data[column+'_tokens'])
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: x.replace('-', " "))
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: x.split(" "))
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: [mot for mot in x if mot])
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: [mot for mot in x  if (mot not in stopword_list)])
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: [mot for mot in x  if (mot != '``')])
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: [stemmer.stem(mot) for mot in x])
    return data
