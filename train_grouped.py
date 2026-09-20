"""Train the three classifiers with a leak-free, group-aware split.

Run make_groups.py first. Every augmented copy of a source photo stays on the
same side of the split (train / validation / test), checkpoints are chosen on
the validation set, and the numbers printed at the end come from the test set
the models never saw. Weights are saved as model/grouped_<name>.weights.h5 and
the old model/*.hdf5 files are not touched.

Run: .venv/bin/python train_grouped.py
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (GRU, LSTM, Bidirectional, Conv2D, Dense, Flatten, MaxPooling2D,
                                     RepeatVector)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'model'
LABELS = ('Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Tiger')
SEED = 42
EPOCHS = {'cnn': 15, 'vgg_bilstm': 20, 'cnn_bigru': 20}
VGG_WEIGHTS = None if os.environ.get('VGG_WEIGHTS', 'imagenet').lower() == 'none' else 'imagenet'
EPOCH_OVERRIDE = int(os.environ.get('EPOCHS', 0))


def load_data():
    X = np.load(MODEL_DIR / 'grouped_X.npy').astype('float32') / 255
    Y = np.load(MODEL_DIR / 'grouped_Y.npy')
    G = np.load(MODEL_DIR / 'grouped_G.npy')
    return X, Y, G


def group_split(X, Y, G):
    """~65% train / ~15% validation / ~20% test, split by source-photo group."""
    idx = np.arange(len(Y))
    trainval, test = next(GroupShuffleSplit(1, test_size=0.20, random_state=SEED).split(idx, Y, G))
    train, val = next(GroupShuffleSplit(1, test_size=0.15 / 0.80, random_state=SEED).split(trainval, Y[trainval], G[trainval]))
    train, val = trainval[train], trainval[val]
    for a, b in ((train, val), (train, test), (val, test)):
        assert not set(G[a]) & set(G[b]), 'a source group leaked across splits'
    return train, val, test


def build_cnn(shape, classes):
    return Sequential([
        Conv2D(32, (3, 3), input_shape=shape, activation='relu'), MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
        Flatten(), Dense(256, activation='relu'), Dense(classes, activation='softmax'),
    ])


def build_vgg_bilstm(shape, classes):
    """Identical layer order to classify._build_model so the weights load there."""
    vgg = VGG19(include_top=False, weights=VGG_WEIGHTS, input_shape=shape)
    vgg.trainable = False
    return Sequential([
        vgg,
        Conv2D(32, (1, 1), activation='relu'), MaxPooling2D(pool_size=(1, 1)),
        Conv2D(32, (1, 1), activation='relu'), MaxPooling2D(pool_size=(1, 1)),
        Flatten(), RepeatVector(2), Bidirectional(LSTM(32)),
        Dense(256, activation='relu'), Dense(classes, activation='softmax'),
    ])


def build_cnn_bigru(shape, classes):
    return Sequential([
        Conv2D(32, (3, 3), input_shape=shape, activation='relu'), MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
        Flatten(), RepeatVector(2), Bidirectional(GRU(32)),
        Dense(256, activation='relu'), Dense(classes, activation='softmax'),
    ])


BUILDERS = {'cnn': build_cnn, 'vgg_bilstm': build_vgg_bilstm, 'cnn_bigru': build_cnn_bigru}


def main():
    X, Y, G = load_data()
    train, val, test = group_split(X, Y, G)
    print(f'images: train {len(train)} | val {len(val)} | test {len(test)}  '
          f'(groups: {len(set(G[train]))}/{len(set(G[val]))}/{len(set(G[test]))})')
    y_cat = to_categorical(Y, len(LABELS))
    results = {}

    for name, build in BUILDERS.items():
        print(f'\n=== {name} ===')
        model = build(X.shape[1:], len(LABELS))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        weights_path = MODEL_DIR / f'grouped_{name}.weights.h5'
        checkpoint = ModelCheckpoint(str(weights_path), monitor='val_loss', save_best_only=True,
                                     save_weights_only=True, verbose=0)
        history = model.fit(X[train], y_cat[train], batch_size=32, epochs=EPOCH_OVERRIDE or EPOCHS[name],
                            validation_data=(X[val], y_cat[val]), callbacks=[checkpoint], verbose=1)
        with open(MODEL_DIR / f'grouped_{name}_history.pckl', 'wb') as handle:
            pickle.dump(history.history, handle)

        model.load_weights(str(weights_path))
        predicted = np.argmax(model.predict(X[test], verbose=0), axis=1)
        truth = Y[test]
        results[name] = {
            'test_accuracy': round(float(accuracy_score(truth, predicted)), 4),
            'macro_precision': round(float(precision_score(truth, predicted, average='macro', zero_division=0)), 4),
            'macro_recall': round(float(recall_score(truth, predicted, average='macro', zero_division=0)), 4),
            'macro_f1': round(float(f1_score(truth, predicted, average='macro', zero_division=0)), 4),
            'confusion_matrix': confusion_matrix(truth, predicted, labels=range(len(LABELS))).tolist(),
        }
        print(f'{name}: test accuracy {results[name]["test_accuracy"]:.3f}, macro-F1 {results[name]["macro_f1"]:.3f}')

    (MODEL_DIR / 'grouped_results.json').write_text(json.dumps({'labels': LABELS, 'results': results}, indent=2))
    print('\nTEST-SET RESULTS (no shared source photos with train/val):')
    for name, result in results.items():
        print(f'  {name:11s} acc {result["test_accuracy"]:.3f}  P {result["macro_precision"]:.3f}  '
              f'R {result["macro_recall"]:.3f}  F1 {result["macro_f1"]:.3f}')
    print('Saved to model/grouped_results.json. To use a model in the app, point classify.py WEIGHTS_PATH at '
          'model/grouped_vgg_bilstm.weights.h5.')


if __name__ == '__main__':
    main()
