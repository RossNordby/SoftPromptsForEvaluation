# Soft Prompts For Evaluation

Code associated with the (upcoming) paper, `Soft Prompts For Evaluation: Measuring Conditional Distance of Capabilities`.

## What is this?

[Soft prompts](https://arxiv.org/abs/2104.08691) are a technique for fine-tuning behavior without modifying a model.
Turns out, they're also a great way to evaluate models! Want to know if there exists some input that might
elicit a target behavior? Optimize a soft prompt for it.

The number of soft prompt tokens required to elicit a behavior provide a proxy for
the amount of conditioning required to elicit that behavior. Very small numbers of tokens
implies that the behavior is natural to the model, while requiring tons of tokens, or failing to improve at all,
is evidence that the model doesn't expose that capability in its input interface.

The failure to elicit a behavior isn't a guarantee that the model *can't* do it (the soft prompt optimization might
just have bad hyperparameters, for example), but soft prompts have the advantage of adversarially exploiting
the model's computation. It's hard for a model to fully hide a capability
against a well-tuned optimization process that can treat its internal computation as a white box.

With that as a foundation, this repository's tests try to measure the conditional distance of various capabilities
relative to various models. It also includes a simple extension to soft prompts to make tests easier to set up:
providing some conditions (e.g. player Elos in a chess game) to input models to generate the soft prompt's
input embedding.

## Environment Setup

For all experiments, you'll need the following:

- python (tested with 3.11.5.)
- pytorch (see [here](https://pytorch.org/get-started/locally/) for installation instructions)
- huggingface transformers: `pip install transformers`
- huggingface accelerate: `pip install accelerate`
- tensorboard: `pip install tensorboard`

For language experiments, you'll also need:

- huggingface datasets: `pip install datasets`

For chess experiments, you'll also need:

- python-chess: `pip install python-chess`
- zstandard: `pip install zstandard`
- A lichess database. You can download one from [here](https://database.lichess.org/).
  The paper uses the November 2023 database.
  `test_shared.py` contains the default path used by chess experiments.

Experiments have been tuned to run on a single GPU with 24 GB of memory.
Many smaller tests will work with less than 16 GB.

## Running Experiments

To run a test, use `python -m tests.<test_name>`.

The `tests` directory contains the code for all experiments. As of this writing, I'm
still in the middle of tuning and there are a lot of tests that won't appear in the paper.

The tests most likely to appear in the paper (in some likely modified form) are:

- `language_autoregressive_test`: Trains a soft prompt on the default pretrained objective.
- `language_skip_tokens_test`: Trains a soft prompt to make the model predict future tokens instead.
- `language_detune_test`: Trains a soft prompt to revert chat fine-tuning.
- `language_trivial_repetition_test`: Trains a soft prompt to repeat the same string over and over.
- `chess_mlp_test`: Trains a soft prompt to predict the next move in a chess game, conditional on player Elos.
- `pathfinding_test`: Trains a soft prompt to predict the next move in a pathfinding game, conditional on path quality.

## Models and datasets

Language and chess experiments mostly use the [EleutherAI pythia suite](https://github.com/EleutherAI/pythia).
The detuning test uses [TinyLlama](https://github.com/jzhang38/TinyLlama).

Language tests use the [RedPajama v2 dataset](https://github.com/togethercomputer/RedPajama-Data).
Chess uses a [lichess database](https://database.lichess.org/) specified in `test_shared.py`.

Pathfinding uses a procedural dataset.

## Hey, where's the paper?

It's not published yet! Hold your horses!