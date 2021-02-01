import os
from pathlib import Path
import numpy as np
import pandas as pd
import spacy
from spacy.compat import pickle
import lz4.frame
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from ehr_classification.tokenizer import get_features, get_custom_tokenizer
from ehr_classification.classifier_model import compile_lstm


def run_multiple_models(df,
                        features,
                        weights,
                        word_vectors,
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

    nlp = get_custom_tokenizer(word_vectors)
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
                               word_vectors,
                               overwrite=False,
                               **kwargs
                               ):
    # only run when not already there
    outfile = Path(outfile)
    if not outfile.exists() or overwrite:
        outfile.touch()
        from .utils import lz4_load
        data_dict = lz4_load(infile)
        predictions = run_multiple_models(df=data_dict['meta'],
                                          features=data_dict['data'],
                                          word_vectors=word_vectors,
                                          **kwargs)
        print('Writing to file')
        predictions.to_parquet(outfile)
        print('Done writing to file')


def run_multiple_models_parquet(infile,
                                outfile,
                                word_vectors,
                                note_column='NoteTXT',
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
        print('Extracting tokens')
        tokenizer = get_custom_tokenizer(word_vectors)
        note_texts = eval_data[note_column]
        tokens = list(tokenizer.pipe(note_texts))
        print('Extracting features')
        eval_features = get_features(tokens, max_note_length)

    eval_data = select_rows(eval_data)
    eval_data = run_multiple_models(df=eval_data,
                                    features=eval_features,
                                    word_vectors=word_vectors,
                                    **kwargs)
    print('Writing to file')
    eval_data.to_parquet(outfile)
    print('Done writing to file')


def run_current_models(infile, outfile, classifier_type, input_type='parquet', **kwargs):
    # use models and vectors path from environment (or use defaults)
    models_path = os.getenv("PREDICT_EHR_MODELS")
    if not models_path:
        models_path = '/mnt/obi0/phi/ehr/models/'
    vectors_path = os.getenv("PREDICT_EHR_VECTORS")
    if not vectors_path:
        vectors_path = '/mnt/obi0/phi/ehr/word_vectors/filtered_20-05-23.bigram'

    if classifier_type == 'event':
        weights = {
            'Event_PCI': f'{models_path}/Events/PCI/LSTM_CNN_BEST_model.hdf5',
            'Event_ACS': f'{models_path}/Events/ACS/LSTM_CNN_BEST_model.hdf5',
            'Event_HF': f'{models_path}/Events/HF/LSTM_CNN_BEST_model.hdf5',
            'Event_IS': f'{models_path}/Events/IS/LSTM_CNN_BEST_model.hdf5'
        }
    elif classifier_type == 'history':
        weights = {
            'History_CAD': f'{models_path}/History/CAD/LSTM_CNN_BEST_model.hdf5',
            'History_CAD_UI': f'{models_path}/History/CAD_UI/LSTM_CNN_BEST_model.hdf5',
            'History_HF': f'{models_path}/History/HF/LSTM_CNN_BEST_model.hdf5',
            'History_HF_UI': f'{models_path}/History/HF_UI/LSTM_CNN_BEST_model.hdf5',
        }
    else:
        raise NotImplementedError

    print(f'Predicting using weights: {weights}')

    if input_type == 'parquet':
        run_multiple_models_parquet(infile=infile,
                                    outfile=outfile,
                                    weights=weights,
                                    word_vectors=vectors_path,
                                    **kwargs)
    elif input_type == 'pickle':
        run_multiple_models_pickle(infile=infile,
                                   outfile=outfile,
                                   weights=weights,
                                   word_vectors=vectors_path,
                                   **kwargs)


def predict(output_directory,
            classifier_type: ('note classifier, `event` or `history`', 'option', 't') = 'event',
            gpu: ('gpu to use', 'option', 'g') = 0,
            gpu_offset: ('subtract gpu offset', 'option', 's') = 0,
            input_type: ('input type, can be `parquet` or `pickle`', 'option', 'i') = 'parquet',
            *file_names):
    """Takes one or more parquet files and writes tokenized text to output file.

    # set environment variables for models and word vectors
    export PREDICT_EHR_VECTORS=en_core_web_lg
    export PREDICT_EHR_MODELS=PATH/TO/MODELS

    # run predictions for events on one or more parquet files
    predict_ehr -t event out_dir text1.parquet
    predict_ehr -t event out_dir text1.parquet text2.parquet text3.parquet

    # run on multiple files in parallel with 4 gpus, using text that has been tokenized before:
    'parallel -j 4 predict_ehr . -t event -g {%} -s 1 -i pickle {} ::: *.pic.lz4'
    'parallel -j 4 predict_ehr . -t history -g {%} -s 1 -i pickle {} ::: *.pic.lz4'

    """

    print(f'Predicting with the following input files: {file_names}')
    for infile in file_names:
        input_file = Path(infile)
        assert Path(output_directory).exists()
        output_file = Path(output_directory) / (input_file.name + '.predictions.pq')
        print('Processing', infile)
        run_current_models(infile,
                           str(output_file),
                           classifier_type=classifier_type,
                           gpu_device=int(gpu) - int(gpu_offset),
                           input_type=input_type)


def predict_():
    """Entry point for console_scripts
    """
    import plac
    plac.call(predict)


def train_model(train_texts,
                train_labels,
                validation_texts,
                validation_labels,
                model_name,
                output_path='.',
                max_note_length=2000,
                learning_rate=0.0001,
                epochs=150,
                batch_size=64,
                gpu_device='0',
                save_best_only=True,
                **kwargs):
    """
    Train a model with train_texts and train_labels and validate on validation_texts and validation_labels.
    train_texts: array of notes to be used for model training.
    train_labels: a binary label to be used for training. The index should correspond to the train_texts
    """

    # use specified gpu device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    # use word vectors from environment variable (or defaults)
    vectors_path = os.getenv("PREDICT_EHR_VECTORS")
    if not vectors_path:
        vectors_path = '/mnt/obi0/phi/ehr/word_vectors/filtered_20-05-23.bigram'
    nlp = get_custom_tokenizer(vectors_path)
    embeddings = nlp.vocab.vectors.data

    print('Parsing texts...')
    train_docs = list(nlp.pipe(train_texts, batch_size=2000))
    validation_docs = list(nlp.pipe(validation_texts, batch_size=2000))
    train_x = get_features(train_docs, max_note_length)
    validation_x = get_features(validation_docs, max_note_length)

    train_labels = [train_labels, train_labels]
    validation_labels = [validation_labels, validation_labels]
    model = compile_lstm(embeddings, {'max_length': max_note_length}, {'lr': learning_rate})

    # define callbacks

    checkpoint_file = model_name + '_{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint_path = os.path.join(output_path, 'checkpoints', checkpoint_file)
    print(f'Saving checkpoints to {checkpoint_path}')
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss', save_best_only=save_best_only, save_weights_only=True
    )
    tensorboard_path = os.path.join(output_path, 'tensorboard', model_name)
    print(f'Writing tensorboard output to {tensorboard_path}')
    tensorboard_callback = TensorBoard(
        log_dir=tensorboard_path,
        write_graph=False, profile_batch=0
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=50)

    print('Training...')
    model.fit(train_x,
              train_labels,
              validation_data=(validation_x, validation_labels),
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[checkpoint_callback, tensorboard_callback, early_stopping_callback])
    return model


def train(labels_path, model_name, output_path,
          epochs: ('number of epochs', 'option', 'e') = 150,
          gpu: ('gpu to use', 'option', 'g') = 0,
          gpu_offset: ('subtract gpu offset', 'option', 's') = 0,
          testrun: ('do short testrun on 200 samples', 'flag', 't') = False,
          all_checkpoints: ('save all or best checkpoint only', 'flag', 'a') = False):
    """Basic training method that takes parquet file with labeled data, splits into training and validation set
     and trains model (with early stopping).

    # first configure a spacy model to use as word vector mapping
    export PREDICT_EHR_VECTORS=en_core_web_lg
    # then train a classifier model given labels
    train_ehr --gpu 0 mgb_predictions_event/Event_PCI_labels.parquet Event_PCI mimic_models_event
    """
    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True)
    print('Processing', labels_path)
    labels_df = pd.read_parquet(labels_path)
    # shuffle the labels
    labels_df = labels_df.sample(frac=1, random_state=42)
    if testrun:
        labels_df = labels_df.iloc[:100]
    # split into two sets for training and validation
    train_df, validation_df = np.array_split(labels_df, 2)
    print(f'Train data shape: {train_df.shape}')
    print(f'Validation data shape: {validation_df.shape}')
    print(f'Training model: {model_name}')
    model = train_model(train_texts=train_df['NoteTXT'],
                        train_labels=train_df['label'],
                        validation_texts=validation_df['NoteTXT'],
                        validation_labels=validation_df['label'],
                        model_name=model_name,
                        output_path=output_path,
                        epochs=int(epochs),
                        save_best_only=not all_checkpoints,
                        gpu_device=int(gpu) - int(gpu_offset))
    model.save_weights(os.path.join(output_path, model_name + '.hdf5'))


def train_():
    """Entry point for console_scripts
    """
    import plac
    plac.call(train)


if __name__ == "__main__":
    predict_()
