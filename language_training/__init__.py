"""
This module contains the functionality for training soft prompts on language related tasks.
"""

from .autoregressive_baseline import AutoregressiveBaseline
from .favor_future_predictions import FavorFutureTest
from .persistent_batch_loader import *
from .trivial_repetition import TrivialRepetitionTest
from .train_and_test_language import *
