import argparse
import pandas as pd
from src.utils import dump_pickle, load_pickle
from src.functions import get_data, generate_corpus_embedding, get_similar_sentences, get_tf_idf, get_similar_sentences_tf_idf
from src.data_processing import data_processing


def run():
    parser = argparse.ArgumentParser(description='Trouver la phrase la plus proche.')
    parser.add_argument("--stage", type = str, help='Stage to run', choices = ['get_data', 'generate_corpus_embedding', 'get_similar_sentences', 'tf_idf_corpus', 'get_similar_sentences_tf_idf'])
    args = parser.parse_args()

    if args.stage == 'get_data':
        input_file_path = 'data/train.json'
        get_data(input_file_path)

    if args.stage == 'tf_idf_corpus':
        data = load_pickle('data/df_fquad.pickle')
        data['question_reponse_concat'] = data['question'] + ' '  + data['reponse']
        data = data_processing(data, 'question_reponse_concat')
        data, tfidf, embedding_tf_df_tensor = get_tf_idf(data)
        dump_pickle(data, 'data/df_fquad_tf_idf.pickle')
        dump_pickle(tfidf, 'data/tf_idf.pickle')
        dump_pickle(embedding_tf_df_tensor, 'data/embedding_tf_idf_tensor.pickle')

    if args.stage == 'generate_corpus_embedding':
        generate_corpus_embedding()

    if args.stage == 'get_similar_sentences':
        queries = ['Coût d\'un crédit', 'Ecart LOSC et premier du campionnat ?', 'Quel type de services propose Agesa ?']
        get_similar_sentences(queries)

    if args.stage == 'get_similar_sentences_tf_idf':
        queries = ['Coût d\'un crédit', 'Ecart LOSC et premier du championnat ?', 'Quel type de services propose Agesa ?']
        get_similar_sentences_tf_idf(queries)



# Run the main
if __name__ == '__main__':
    run()
