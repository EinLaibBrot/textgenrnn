@@ -1,31 +1,48 @@

from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
# https://github.com/keras-team/keras/issues/14440#issuecomment-869113052
try:
    # on Tensorflow < 2.5.0
    from tensorflow.keras.utils import multi_gpu_model
except ImportError:
    try:
        # on Tensorflow >= 2.5.0
        from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
    except:
        raise

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow import config as config
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf

# https://stackoverflow.com/a/69006914
try:
    # https://stackoverflow.com/a/63752287
    # on Keras == 2.2.0
    from keras.backend.tensorflow_backend import set_session
except ModuleNotFoundError:
    try:
        from keras.backend import set_session
    except:
        raise

import numpy as np
import json
import h5py
from pkg_resources import resource_filename
from .model import textgenrnn_model

from .model_training import *
from .utils import *
import csv
import re


class textgenrnn:
 @@ -57,9 +74,9 @@ def __init__(self, weights_path=None,
                                           'textgenrnn_vocab.json')

        if allow_growth is not None:
            c = tf.ConfigProto()
            c.gpu_options.allow_growth = True
            set_session(tf.Session(config=c))

        if config_path is not None:
            with open(config_path, 'r',
@@ -86,7 +103,7 @@ def generate(self, n=1, return_as_list=False, prefix=None,
                 max_gen_length=300, interactive=False,
                 top_n=3, progress=True):
        gen_texts = []
        iterable = trange(n) if progress and n > 1 else range(n)
        for _ in iterable:
            gen_text, _ = textgenrnn_generate(self.model,
                                              self.vocab,
@@ -145,6 +162,9 @@ def train_on_texts(self, texts, context_labels=None,
        if context_labels:
            context_labels = LabelBinarizer().fit_transform(context_labels)
        if 'prop_keep' in kwargs:
            train_size = prop_keep

        if self.config['word_level']:
            # If training word level, must add spaces around each
            # punctuation. https://stackoverflow.com/a/3645946/931441
