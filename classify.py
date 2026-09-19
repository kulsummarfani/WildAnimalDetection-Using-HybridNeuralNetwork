"""Second-stage wildlife species classifier for YOLO detection crops."""

from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Bidirectional, Conv2D, Dense, Flatten, LSTM, MaxPooling2D, RepeatVector
from tensorflow.keras.models import Sequential

LABELS = ('Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Tiger')
IMG_SIZE = 32
WEIGHTS_PATH = Path(__file__).resolve().parent / 'model' / 'vgg_bilstm_weights.hdf5'

_model = None


def _build_model(num_classes: int) -> Sequential:
    """Recreate the architecture used to produce vgg_bilstm_weights.hdf5."""
    # All VGG weights are stored in the supplied HDF5 checkpoint. Using None
    # keeps inference fully offline and avoids an unnecessary ImageNet download.
    vgg = VGG19(include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    vgg.trainable = False

    model = Sequential([
        vgg,
        Conv2D(32, (1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        Conv2D(32, (1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        Flatten(),
        RepeatVector(2),
        Bidirectional(LSTM(32)),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    return model


def load_classifier() -> Sequential:
    """Build and cache the classifier once for the current Python process."""
    global _model
    if _model is None:
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(f'Classifier weights not found: {WEIGHTS_PATH}')
        _model = _build_model(len(LABELS))
        _model.load_weights(WEIGHTS_PATH)
    return _model


def predict_species(crop_bgr: np.ndarray) -> tuple[str, float]:
    """Return the predicted species and confidence for a non-empty OpenCV crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError('Cannot classify an empty detection crop.')

    image = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    probabilities = load_classifier().predict(np.expand_dims(image, axis=0), verbose=0)[0]
    index = int(np.argmax(probabilities))
    return LABELS[index], float(probabilities[index])
