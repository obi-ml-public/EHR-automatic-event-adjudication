#!/usr/bin/env python
# coding: utf8

import spacy
from pathlib import Path
from spacy.symbols import ORTH, LOWER
from tqdm.auto import tqdm
import pandas as pd


def read_abbreviations():
    try:
        import importlib.resources as pkg_resources
    except ImportError:
        # Try backported to PY<37 `importlib_resources`.
        import importlib_resources as pkg_resources
    from . import abbreviations  # relative-import the *package* containing the templates

    abbrevs = []
    with pkg_resources.open_text(abbreviations, 'medical_abbreviations_curated.txt') as f:
        abbrevs += [line.rstrip('\n') for line in f]
    with pkg_resources.open_text(abbreviations, 'medical_abbreviations_wiki.txt') as f:
        abbrevs += [line.rstrip('\n') for line in f]

    return abbrevs


# read in tokenizer exceptions
def get_custom_tokenizer(model):
    print(f"Loading Spacy model: {model}")
    nlp = spacy.load(model)
    abbrevs = read_abbreviations()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    # this matches with capitalization
    exclusions_cased = {abbreviation: [{ORTH: abbreviation}] for abbreviation in abbrevs}
    for k, excl in exclusions_cased.items():
        tokenizer.add_special_case(k, excl)

    # TODO can't easily match uncased, only lower case
    exclusions_uncased = {abbreviation.lower(): [{ORTH: abbreviation.lower()}] for abbreviation in abbrevs}
    for k, excl in exclusions_uncased.items():
        try:
            tokenizer.add_special_case(k, excl)
        except:
            print('failed to add exception: {}'.format(k))
    return tokenizer


def get_features(docs, max_length):
    import numpy as np
    docs = list(docs)
    features = np.zeros((len(docs), max_length), dtype='int32')
    for i, doc in tqdm(enumerate(docs)):
        j = 0
        for token in doc:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                features[i, j] = vector_id
            else:
                features[i, j] = 0
            j += 1
            if j >= max_length:
                break
    return features


# less general approach for faster preprocessing
def tokenize_and_map(input_file, word_vectors, maxlen):
    from .utils import lz4_dump
    tokenizer = get_custom_tokenizer(word_vectors)
    input_df = pd.read_parquet(input_file, columns=['PatientID', 'NoteID', 'ContactDTS', 'NoteTXT'])
    print('Tokenizing')
    docs = [doc for doc in tqdm(list(tokenizer.pipe(input_df.NoteTXT, batch_size=10000)))]
    print('Mapping to features')
    features = get_features(docs, maxlen)
    lz4_dump(
        {'meta': input_df[['PatientID', 'NoteID', 'ContactDTS']], 'data': features},
        input_file.replace('notes', 'features').replace('parquet', 'pic.lz4')
    )


def read_input(location):
    if not isinstance(location, Path):
        location = Path(location)
    if location.is_dir():
        for file in location.glob('*.parquet'):
            print(file)
            for text in pd.read_parquet(file, columns=['NoteTXT'])['NoteTXT']:
                yield text
    else:
        for text in pd.read_parquet(location, columns=['NoteTXT'])['NoteTXT']:
            yield text


def main(input_path, output_path,
         binary: ('convert to binary token vector', 'flag', 'b'),
         uncased: ('convert to uncased', 'flag', 'u'),
         maxlen: ('max len for binary vector', 'option', 'l') = 2000,
         word_vectors: ('word vectors', 'option', 'w') = '/mnt/obi0/phi/ehr/word_vectors/filtered_20-05-23.bigram'):
    """Takes parquet file or directory and writes tokenized text to output file.

    # To tokenize into text files (e.g. to train embeddings), do:
    tokenize --word_vectors en_core_web_lg simulated_notes/simulated_notes.parquet simulated_notes/simulated_notes_tokenized.txt

    # To tokenize and map to features (numpy array) use -b flag
    # requires these columns in parquet file: ['PatientID', 'NoteID', 'ContactDTS', 'NoteTXT']
    tokenize --word_vectors en_core_web_lg simulated_notes/simulated_notes.parquet no_out -b

    # To run multiple tokenization processes in parallel do:
    parallel -j 10 tokenize --word_vectors en_core_web_lg -b {} no_out ::: simulated_notes/*.parquet

    """

    if binary:
        tokenize_and_map(input_path, word_vectors, maxlen)
    else:
        # use tokenizer pipe
        tokenizer = get_custom_tokenizer(word_vectors)
        with open(output_path, "a") as f:
            for doc in tqdm(tokenizer.pipe(read_input(input_path), batch_size=10000)):
                if uncased:
                    f.write(" ".join(token.text.lower() for token in doc))
                else:
                    f.write(" ".join(token.text for token in doc))
                f.write("\n")


def main_():
    """Entry point for console_scripts
    """
    import plac
    plac.call(main)


if __name__ == "__main__":
    main_()
