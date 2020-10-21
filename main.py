import argparse
import pandas as pd
from src.utils import dump_pickle, load_pickle
from src.functions import get_data, generate_corpus_embedding, get_similar_sentences


def run():
    parser = argparse.ArgumentParser(description='Trouver la phrase la plus proche.')
    parser.add_argument("--stage", type = str, help='Stage to run', choices = ['get_data', 'generate_corpus_embedding', 'get_similar_sentences'])
    args = parser.parse_args()

    if args.stage == 'get_data':
        input_file_path = 'data/train.json'
        get_data(input_file_path)

    if args.stage == 'generate_corpus_embedding':
        generate_corpus_embedding()

    if args.stage == 'get_similar_sentences':
        queries = ['Coût d\'un crédit', 'Ecart LOSC et premier du campionnat ?', 'Quel type de services propose Agesa ?']
        get_similar_sentences(queries)

# Run the main
if __name__ == '__main__':
    run()
