import os
from pathlib import Path
import pandas as pd
import spacy
from spacy.compat import pickle
import lz4.frame
from tqdm import tqdm

from ehr_classification.classifier_model import compile_lstm


def run_multiple_models(df,
                        features,
                        weights,
                        word_vectors='/mnt/obi0/phi/ehr/word_vectors/filtered_20-05-23.bigram',
                        max_note_length=2000,
                        batch_size=64,
                        gpu_device='0'
                        ):
    '''
    Run model on infile, adds columns for predictions and save it to outfile
    :param df:
    :param features:
    :param weights:
    :param word_vectors:
    :param max_note_length:
    :param batch_size:
    :param gpu_device:
    :return:
    '''

    # use specified gpu device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    nlp = spacy.load(word_vectors)
    embeddings = nlp.vocab.vectors.data
    model = compile_lstm(embeddings,
                         {'nr_hidden': 64, 'max_length': max_note_length, 'nr_class': 4},
                         {'dropout': 0.5, 'lr': 0.0001})

    for target, weight in tqdm(list(weights.items())):
        model.load_weights(weight)
        print(f'Predicting {target}.')
        predictions = model.predict(features, batch_size=batch_size, verbose=True)
        print(f'Done predicting {target}.')
        df[(target + '_predictions')] = predictions[0]
        df[(target + '_raw')] = predictions[1]
    return df


def run_multiple_models_pickle(infile,
                               outfile,
                               overwrite=False,
                               **kwargs
                               ):

    # only run when not already there
    outfile = Path(outfile)
    if not outfile.exists() or overwrite:
        outfile.touch()
        from .utils import lz4_load
        data_dict = lz4_load(infile)
        predictions = run_multiple_models(data_dict['meta'], data_dict['data'], **kwargs)
        print('Writing to file')
        predictions.to_parquet(outfile)
        print('Done writing to file')


def run_multiple_models_parquet(infile,
                                outfile,
                                note_column='NoteTXT',
                                word_vectors='/mnt/obi0/phi/ehr/word_vectors/filtered_20-05-23.bigram',
                                max_note_length=2000,
                                **kwargs
                                ):
    def select_rows(df):  # Remove rows with empty note text
        df = pd.DataFrame(df.loc[df[note_column].notnull()])
        return df

    eval_data = pd.read_parquet(infile)
    lz4_file = infile.replace('.parquet', '.pic.lz4')
    if Path(lz4_file).exists():
        print('Loading features')
        with lz4.frame.open(lz4_file, mode='r') as f:
            eval_docs = pickle.load(f)
    else:
        from ehr_classification.tokenizer import get_features, get_custom_tokenizer
        print('Extracting tokens')
        tokenizer = get_custom_tokenizer(word_vectors)
        note_texts = eval_data[note_column]
        tokens = list(tokenizer.pipe(note_texts))
        print('Extracting features')
        eval_docs = get_features(tokens, max_note_length)

    eval_data = select_rows(eval_data)
    eval_data = run_multiple_models(eval_data, eval_docs, **kwargs)
    print('Writing to file')
    eval_data.to_parquet(outfile)
    print('Done writing to file')


def run_current_models(infile, outfile, classifier_type, input_type='parquet', **kwargs):
    if classifier_type == 'event':
        weights = {
            'Event_PCI': '/mnt/obi0/phi/ehr/models/Events/PCI/LSTM_CNN_BEST_model.hdf5',
            'Event_ACS': '/mnt/obi0/phi/ehr/models/Events/ACS/LSTM_CNN_BEST_model.hdf5',
            'Event_HF': '/mnt/obi0/phi/ehr/models/Events/HF/LSTM_CNN_BEST_model.hdf5',
            'Event_IS': '/mnt/obi0/phi/ehr/models/Events/IS/LSTM_CNN_BEST_model.hdf5'
        }
    elif classifier_type == 'history':
        weights = {
            'History_CAD': '/mnt/obi0/phi/ehr/models/History/CAD/LSTM_CNN_BEST_model.hdf5',
            'History_CAD_UI': '/mnt/obi0/phi/ehr/models/History/CAD_UI/LSTM_CNN_BEST_model.hdf5',
            'History_HF': '/mnt/obi0/phi/ehr/models/History/HF/LSTM_CNN_BEST_model.hdf5',
            'History_HF_UI': '/mnt/obi0/phi/ehr/models/History/HF_UI/LSTM_CNN_BEST_model.hdf5',
        }
    else:
        raise NotImplementedError

    if input_type == 'parquet':
        run_multiple_models_parquet(infile=infile,
                                    outfile=outfile,
                                    weights=weights,
                                    **kwargs)
    elif input_type == 'pickle':
        run_multiple_models_pickle(infile=infile,
                                   outfile=outfile,
                                   weights=weights,
                                   **kwargs)


def main(output_directory,
         classifier_type: ('note classifier, `event` or `history`', 'option', 't') = 'event',
         gpu: ('gpu to use', 'option', 'g') = 0,
         gpu_offset: ('subtract gpu offset', 'option', 's') = 0,
         input_type: ('input type, can be `parquet` or `pickle`', 'option', 'i') = 'parquet',
         *file_names):
    """Takes feather file or directory and writes tokenized text to output file.
    predict_ehr -t event text.parquet

    Run in parallel with 4 gpus on a single machine, using text that has been tokenized before:
    'parallel -j 4 predict_ehr . -t event -g {%} -s 1 -i pickle {} ::: *.pic.lz4'
    'parallel -j 4 predict_ehr . -t history -g {%} -s 1 -i pickle {} ::: *.pic.lz4'

    """

    for infile in file_names:
        input_file = Path(infile)
        assert Path(output_directory).exists()
        output_file = Path(output_directory) / (input_file.name + '.predictions.pq')
        print('Processing', infile)
        run_current_models(infile,
                           str(output_file),
                           classifier_type=classifier_type,
                           gpu_device=int(gpu)-int(gpu_offset),
                           input_type=input_type)


# TODO console entry point not working with plac
def run():
    """Entry point for console_scripts
    """
    import plac
    plac.call(main)


if __name__ == "__main__":
    run()
