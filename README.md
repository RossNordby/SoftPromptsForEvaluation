# Soft Prompts For Evaluation

Code associated with the (upcoming) paper, [Soft Prompts For Evaluation: Measuring Conditional Distance of Capabilities](https://1drv.ms/b/s!AkbNRQZiRerJ60FS9KJ1qqIo9qyC?e=2AqBeH).

The onedrive link used at the time of writing will likely expire. Ideally, it will be replaced with a complete version on arxiv soonishly.

## What is this?

[Soft prompts](https://arxiv.org/abs/2104.08691) are a technique for fine-tuning behavior without modifying a model.
Turns out, they're also a great way to evaluate models!
 - Want to know if there exists some input that might jailbreak your model's friendliness? 
 - Need to know whether your model has the ability to self-exfiltrate?
 - Curious whether your model has easily-accessed latent omnicidal tendencies? 

Optimize a soft prompt for it! (Or, ideally, for a proxy behavior that doesn't involve omnicide.)

The number of soft prompt tokens required to elicit a behavior provide a proxy for
the amount of conditional information required to elicit that behavior. Very small numbers of tokens
implies that the behavior is natural to the model, while requiring tons of tokens, or failing to improve at all,
is evidence that the model doesn't expose that capability in its input interface.

The failure to elicit a behavior isn't a guarantee that the model *can't* do it (the soft prompt optimization might
just have bad hyperparameters, for example), but soft prompts have the advantage of adversarially exploiting
the model's computation. It's hard for a model to fully hide a capability
against a well-tuned optimization process that can treat its internal computation as a white box.

With that as a foundation, this repository's tests try to measure the distance to "conditional saturation" for various capabilities
relative to various models. The compute budget for this paper is too limited for omnicide proxies,
but the underlying technique can be demonstrated!

It also includes a simple extension to soft prompts to make tests easier to set up:
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
The paper uses the November 2023 database for training and the December 2023 database
for evaluation. Yes, that's an extra 31 gigabytes for a trivial amount of evaluation;
sorry about that. You can swap it out, though! See the `chess_database_path` and
`chess_test_database_path` in `chess_test.py`.

Experiments have been tuned to run on a single GPU with 24 GB of memory.
Many smaller tests will work with less than 16 GB.
Don't expect distributed training to work out of the box; I wasn't able to
get around to testing it.

## Running Experiments

To run a test, use `python -m tests.<test_name>`.

The `tests` directory contains the code for all experiments. 

The tests that appear in the paper are:

- `autoregressive_test`: Trains a soft prompt on the default pretrained objective.
- `skip_tokens_test`: Trains a soft prompt to make the model predict future tokens instead.
- `chat_detuning_test`: Trains a soft prompt to revert chat fine-tuning.
- `repetition_test`: Trains a soft prompt to repeat the same string over and over.
- `chess_test`: Trains a soft prompt to predict the next move in a chess game, conditional on player Elos.
- `pathfinding_test`: Trains a soft prompt to predict the next move in a pathfinding game, conditional on path quality.

All tests record information with tensorboard into the `runs` directory.
Upon completing a training run, a snapshot and result will be recorded
into `snapshots` and `results` respectively.

Individual training runs are *relatively* small, typically taking no more than a couple of hours.
There is no built-in mid-training checkpointing. Yes, that's a bit annoying and 
lost me several hours of compute, sorry about that; I've been operating at approximately 
maximum busy for the last two months. Maybe later!

The `exploratory_tests` directory contains a bunch of other tests.
These were primarily used to guide the development of the main tests and to perform diagnostics.
At the time of writing, many will not run in their current form.

## Results

The `results` directory contains the results of the tests that appear in the paper.
The `runs` directory contains the tensorboard logs for these tests.

Snapshots are too large to include in this repository. Running the tests will reproduce them.
You can also contact me; it's a 1.2GB zip file.

## Models and datasets

Language and chess experiments mostly use the [EleutherAI pythia suite](https://github.com/EleutherAI/pythia).
The detuning test uses [TinyLlama](https://github.com/jzhang38/TinyLlama).

Language tests use the [RedPajama v2 dataset](https://github.com/togethercomputer/RedPajama-Data) and the [Pile](https://arxiv.org/abs/2101.00027).
Chess uses [lichess databases](https://database.lichess.org/).

Pathfinding uses a procedural dataset.