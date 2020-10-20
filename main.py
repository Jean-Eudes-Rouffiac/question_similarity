import argparse
import pandas as pd
from src.data_processing import get_FQuAD_data
from src.utils import dump_pickle, load_pickle
import json
from sentence_transformers import SentenceTransformer, util
from transformers import CamembertModel, CamembertTokenizer
import torch


def get_data(input_file_path):
    with open(input_file_path ,mode="rt",encoding="utf-8") as file:
        qa_data = json.load(file)
    data = get_FQuAD_data(qa_data)
    dump_pickle(data, 'data/df_fquad.pickle')
    print(data.head(10))

def get_top_5_questions_bert_multi():
    embedder = SentenceTransformer('distiluse-base-multilingual-cased')
    # Corpus with example sentences
    corpus = ['Quelle est la différence entre le net à payer et le net imposable ?',
              'Quelle est la base de calcul de la CSG et CRDS ? ',
              'Quelle est la différence entre les jours ouvrés, les jours non ouvrés, les jours calendaires ? ',
              'un particulier a-t\'il le droit la possibilité de facturer une entreprise?',
              'Je dois facturer un service que mon association a rendu a une société de spectacle.  Dois-je faire apparaître le prix HT ? Dois-je me faire payer directement le prix HT ? ',
              'Différence entre enregistrer une facture en 408100 -FNP- et 468600 -charges à payer » ? ',
              'Que sont les effets escomptés non échus ?',
              'Sur quel compte sont comptabilisés les frais de déchetterie facturés aux entreprises ?',
              'Quelle est la différence entre une filiale et une succursale ? '
              ]
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    # Query sentences:
    queries = ['Différence entre une filiale et une succursale', 'Comment facturer un service rendue par une association ?', 'Qu\'est ce qu\'est le net à payer ?']


    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = 5
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()

        #We use torch.topk to find the highest 5 scores
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            print(corpus[idx], "(Score: %.4f)" % (score))



def run():
    parser = argparse.ArgumentParser(description='Trouver la phrase la plus proche.')
    parser.add_argument("--stage", type = str, help='Stage to run', choices = ['get_data', 'get_top_5_questions_bert_multi', 'get_top_5_questions_camembert'])
    args = parser.parse_args()

    if args.stage == 'get_data':
        input_file_path = 'data/train.json'
        get_data(input_file_path)

    if args.stage == 'get_top_5_questions_bert_multi':
        get_top_5_questions_bert_multi()

    if args.stage == 'get_top_5_questions_camembert':
        get_top_5_questions_camembert()


# Run the main
if __name__ == '__main__':
    run()
