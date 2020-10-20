import pandas as pd
import json
import numpy as np

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
