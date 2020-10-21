import json
import torch
from src.data_processing import get_FQuAD_data
from src.utils import dump_pickle, load_pickle
from sentence_transformers import SentenceTransformer, util
from transformers import CamembertModel, CamembertTokenizer


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
