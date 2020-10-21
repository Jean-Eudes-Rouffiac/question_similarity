import json
import torch
import pandas as pd
from src.data_processing import get_FQuAD_data, data_processing
from src.utils import dump_pickle, load_pickle
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def get_data(input_file_path):
    with open(input_file_path ,mode="rt",encoding="utf-8") as file:
        qa_data = json.load(file)
    data = get_FQuAD_data(qa_data)
    dump_pickle(data, 'data/df_fquad.pickle')

def generate_corpus_embedding():
    data = load_pickle('data/df_fquad.pickle')
    embedder = SentenceTransformer('distiluse-base-multilingual-cased')
    corpus_embeddings = embedder.encode(list(data['question']), convert_to_tensor=True)
    dump_pickle(corpus_embeddings, 'data/fquad_questions_embedding.pickle')

def get_similar_sentences(queries, top_k = 5):
    data = load_pickle('data/df_fquad.pickle')
    corpus_embeddings = load_pickle('data/fquad_questions_embedding.pickle')
    embedder = SentenceTransformer('distiluse-base-multilingual-cased')

    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()

        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            print(list(data['question'])[idx], "(Score: %.4f)" % (score))

def get_tf_idf(data):
    data['question_reponse_concat_join'] = data['question_reponse_concat_tokens'].apply(lambda x: ' '.join(x))
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(data['question_reponse_concat_join'])
    embedding_tf_df_tensor = torch.tensor(x.toarray())
    return data, tfidf, embedding_tf_df_tensor

def get_similar_sentences_tf_idf(queries, top_k = 5):
    data = load_pickle('data/df_fquad_tf_idf.pickle')
    corpus_embeddings_tf_idf = load_pickle('data/embedding_tf_idf_tensor.pickle')
    tf_idf = load_pickle('data/tf_idf.pickle')

    for query in queries:
        real_query = query
        query_df = pd.DataFrame([query], columns = ['question'])
        query_df = data_processing(query_df, 'question')
        query_df['question_tokens'] = query_df['question_tokens'].apply(lambda x: ' '.join(x))
        query = list(query_df['question_tokens'])
        question_tfidf = torch.tensor(tf_idf.transform(query).toarray())
        cos_scores = util.pytorch_cos_sim(question_tfidf, corpus_embeddings_tf_idf)[0]
        cos_scores = cos_scores.cpu()

        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", real_query)
        print("\nTop 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            print(list(data['question'])[idx], "(Score: %.4f)" % (score))
