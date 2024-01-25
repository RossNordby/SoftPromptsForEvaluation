from typing import Callable, TypeAlias

from accelerate import Accelerator
from torch import Tensor
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import torch

from soft_prompting.prompted_embeddings_builder import build_prompted_embeddings
from soft_prompting import soft_prompts, TaskBatchDataPreparer, SoftPromptLossFunction, EmbedInputFunction
from soft_prompting.batch_loader import BatchLoader
from soft_prompting.data_logger import DataLogger
from soft_prompting.utils import sample_token_from_logits, devices_match, get_token_counts


# GetDataForSoftPromptTrainingFunction: TypeAlias = Callable[
#     [Tensor, Tensor, soft_prompts.SoftPrompt, Tensor, EmbedInputFunction, int, int], tuple[
#         Tensor, Tensor, SoftPromptLossFunction | None]]
# """
# A function of the form:
# (input samples, soft prompt start indices, soft prompt, soft prompt parameters, embedding function,
# end of text token id, pad token id) ->
# (input embeddings, output labels, loss function)
# """


def run_training_batch(model: GPTNeoXForCausalLM, optimizer: torch.optim.Optimizer,
                       accelerator: Accelerator,
                       input_samples: Tensor, input_embeddings: Tensor, output_labels: Tensor | None,
                       soft_prompt_start_indices: torch.Tensor,
                       soft_prompt: soft_prompts.SoftPrompt,
                       compute_loss: SoftPromptLossFunction | None):
    """
    Performs a backward step on the model using the given labels and default loss function, or a given loss function
    is no labels are provided.
    :param model: Model to train.
    :param optimizer: Optimizer used to train the model.
    :param accelerator: Accelerator used to train the model.
    :param input_samples: Original input samples used to train the model.
    :param input_embeddings: Embeddings used to train the model. Contains soft_prompt tokens.
    :param output_labels: Labels used to train the model. Should contain -100 for tokens related to the soft prompt.
    :param soft_prompt_start_indices: Indices of the soft prompt in the input embeddings.
    :param soft_prompt: Soft prompt being trained.
    :param compute_loss: Loss function to use if no labels are provided.
    Parameters are (logits, soft_prompt_start_indices).
    """
    assert devices_match(input_embeddings.device, accelerator.device)
    assert devices_match(input_samples.device, accelerator.device)
    assert output_labels is None or devices_match(output_labels.device, accelerator.device)

    with accelerator.accumulate([model, soft_prompt]):
        if compute_loss is None:
            # If no custom loss was specified, we'll use the default model loss, so we need to provide labels.
            model_outputs = model.forward(inputs_embeds=input_embeddings, labels=output_labels)
            loss = model_outputs.loss
        else:
            model_outputs = model.forward(inputs_embeds=input_embeddings)
            # Labels weren't provided. There must be a loss function defined.
            # (logits, labels, soft prompt, soft prompt start indices) -> loss.
            loss = compute_loss(model_outputs.logits, output_labels, soft_prompt, soft_prompt_start_indices)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()
    return loss


def test_loss(model: GPTNeoXForCausalLM, tokenizer: GPTNeoXTokenizerFast,
              input_samples: Tensor, input_embeddings: Tensor, output_labels: Tensor | None,
              training_step_index: int, soft_prompt_start_indices: torch.Tensor,
              soft_prompt: soft_prompts.SoftPrompt,
              compute_loss: SoftPromptLossFunction | None,
              logger: DataLogger | None = None,
              soft_prompt_string: str | None = None):
    """
    Evaluates the loss of the model and prompt without optimizing. Prints diagnostics.
    :param model: Model to train.
    :param tokenizer: Tokenizer used to decode samples.
    :param input_samples: Original input samples used to train the model.
    :param input_embeddings: Embeddings used to train the model. Contains soft_prompt tokens.
    :param output_labels: Labels used to train the model. Should contain -100 for tokens related to the soft prompt.
    :param training_step_index: Index of the training step.
    :param soft_prompt_start_indices: Indices of the soft prompt in the input embeddings.
    :param soft_prompt: Soft prompt being trained.
    :param compute_loss: Loss function to use if no labels are provided.
    Parameters are (logits, soft_prompt_start_indices).
    :param logger: Logger to log loss to.
    :param soft_prompt_string: String to insert in the debug outputs to mark the start of the soft prompt, if any.
    If None, no string is inserted.
    """
    print(f'Prompt inputs for {training_step_index}')
    print(input_embeddings[0,
          soft_prompt_start_indices[0].item():soft_prompt_start_indices[
                                                  0].item() + soft_prompt.soft_prompt_token_count,
          :8])

    assert devices_match(input_embeddings.device, model.device)
    assert devices_match(input_samples.device, model.device)
    assert output_labels is None or devices_match(output_labels.device, model.device)

    with torch.no_grad():
        if compute_loss is None:
            # If no custom loss was specified, we'll use the default model loss, so we need to provide labels.
            model_outputs = model.forward(inputs_embeds=input_embeddings, labels=output_labels)
            loss = model_outputs.loss
        else:
            model_outputs = model.forward(inputs_embeds=input_embeddings)
            # Labels weren't provided. There must be a loss function defined.
            # (logits, labels, soft prompt, soft prompt start indices) -> loss.
            loss = compute_loss(model_outputs.logits, output_labels, soft_prompt, soft_prompt_start_indices)

    if logger is not None:
        logger.add_scalar('Test loss', loss, training_step_index)

    print(f"Completed test loss for {training_step_index}, loss: {loss}")
    for j in range(1):
        next_token_ids = sample_token_from_logits(model_outputs.logits[j]).squeeze()

        print(f'INPUTS {j}, INSERTED SOFT PROMPT AT {soft_prompt_start_indices[j]}')
        print(tokenizer.decode(input_samples[j]))
        print()
        print(f'OUTPUTS {j}:')
        # Pull out the soft prompt's predictions:
        before_tokens = next_token_ids[:soft_prompt_start_indices[j]]
        after_tokens = next_token_ids[soft_prompt_start_indices[j] + soft_prompt.soft_prompt_token_count:]
        before = tokenizer.decode(before_tokens)
        after = tokenizer.decode(after_tokens)
        print(f'{before}{soft_prompt_string}{after}')
        print()
        print("OUTPUTS FOR SOFT PROMPT TOKENS:")
        print(tokenizer.decode(next_token_ids[soft_prompt_start_indices[j]:
                                              soft_prompt_start_indices[j] + soft_prompt.soft_prompt_token_count]))
        print()
        print("_____")
        print()


def generate(prompt_ids: Tensor, ids_to_embeddings, model, soft_prompt: soft_prompts.SoftPrompt,
             soft_prompt_parameters: Tensor | None, tokenizer,
             batch_size: int, generated_token_count: int) -> Tensor:
    with torch.no_grad():
        soft_prompt_start_indices = get_token_counts(prompt_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)

        raw_embeddings = ids_to_embeddings(prompt_ids)

        initial_input_embeddings = build_prompted_embeddings(raw_embeddings, soft_prompt_start_indices,
                                                             soft_prompt, soft_prompt_parameters)

        output_ids = torch.empty([batch_size, generated_token_count], dtype=torch.long, device=model.device)

        assert devices_match(raw_embeddings.device, model.device)
        assert devices_match(soft_prompt_start_indices.device, model.device)

        next_generated_indices = soft_prompt_start_indices + soft_prompt.soft_prompt_token_count
        previous_key_values = None
        token_ids_to_append = None
        for i in range(generated_token_count):
            # Autoregressively feed the outputs back into the model until we reach the target token count.
            if previous_key_values is None:
                # This is the first iteration; provide the input embeddings.
                model_outputs = model.forward(inputs_embeds=initial_input_embeddings, use_cache=True)
                batch_indices = torch.arange(batch_size, device=model.device).unsqueeze(1)
                gathered_logits = model_outputs.logits[batch_indices, (next_generated_indices - 1).unsqueeze(1)]
                token_ids_to_append = sample_token_from_logits(gathered_logits)
            else:
                # This is a later iteration. We no longer need to supply the full embeddings as input;
                # the last input ids suffice.
                # We sorta shrugged about how this handles padding tokens. The initial execution handled it
                # (because it's a holdover from a much slower but technically more robust implementation),
                # but I would be a bit surprised if partial sequences worked correctly right now. Not a big deal for
                # a pure debug feature.
                model_outputs = model.forward(input_ids=token_ids_to_append, use_cache=True,
                                              past_key_values=previous_key_values)
                token_ids_to_append = sample_token_from_logits(model_outputs.logits)
            output_ids[:, i] = token_ids_to_append.squeeze(1)

            next_generated_indices += 1
            previous_key_values = model_outputs.past_key_values
        return output_ids


def generate_from_batch_loader(batch_loader: BatchLoader, ids_to_embeddings: EmbedInputFunction, model,
                               soft_prompt: soft_prompts.SoftPrompt, tokenizer,
                               generated_token_count: int) -> (Tensor, Tensor):
    """
    Generates tokens using prompts pulled from a batch loader.
    :param batch_loader: Batch loader to pull prompts from.
    :param ids_to_embeddings: Function that converts token ids to embeddings.
    :param model: Model to generate with.
    :param soft_prompt: Soft prompt to insert after the prompt.
    :param tokenizer: Tokenizer to use.
    :param generated_token_count: Number of tokens to generate.
    :return: A tuple of (prompt ids, generated ids).
    """
    soft_prompt_parameters, samples, _ = next(batch_loader)
    if soft_prompt_parameters is not None:
        soft_prompt_parameters = soft_prompt_parameters.to(model.device)
    samples = samples.to(model.device)
    return (samples,
            generate(samples, ids_to_embeddings, model, soft_prompt, soft_prompt_parameters, tokenizer, samples.size(0),
                     generated_token_count))


def create_strings_from_prompted_generation(prompt_ids: Tensor, generated_ids: Tensor, tokenizer,
                                            soft_prompt_string: str | None = None) -> list[str]:
    """
    Creates strings for each lane in a batch from prompt ids and generated ids, optionally including a marker for
    where the soft prompt was inserted.
    :param prompt_ids: Prompt ids with dimensions [batch_size, token_count], Lanes may be right-padded.
    :param generated_ids: Generated ids with dimensions [batch_size, token_count].
    :param tokenizer: Tokenizer used to decode ids.
    :param soft_prompt_string: String to insert to mark the start of the soft prompt, if any. If None, no string is
    inserted.
    :return: A list of strings for the batch.
    """
    token_counts = get_token_counts(prompt_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
    strings = []
    for i in range(prompt_ids.size(0)):
        prompt_string = tokenizer.decode(prompt_ids[i, :token_counts[i]])
        if soft_prompt_string is not None:
            prompt_string += soft_prompt_string
        generated_string = tokenizer.decode(generated_ids[i])
        strings.append(prompt_string + generated_string)
    return strings


def generate_for_forward_testing(batch_loader: BatchLoader, ids_to_embeddings: EmbedInputFunction, model,
                                 soft_prompt: soft_prompts.SoftPrompt, tokenizer, generated_token_count: int,
                                 prompt_string: str | None = None):
    """
    Generates tokens using prompts pulled from a batch loader and prints them.
    :param batch_loader: Batch loader to pull prompts from.
    :param ids_to_embeddings: Function that converts token ids to embeddings.
    :param model: Model to generate with.
    :param soft_prompt: Soft prompt to insert after the prompt.
    :param tokenizer: Tokenizer to use.
    :param generated_token_count: Number of tokens to generate.
    :param prompt_string: String to insert to mark the start of the soft prompt, if any. If None, no string is
    inserted.
    """
    input_ids, output_ids = generate_from_batch_loader(batch_loader, ids_to_embeddings, model, soft_prompt, tokenizer,
                                                       generated_token_count)
    strings = create_strings_from_prompted_generation(input_ids, output_ids, tokenizer, prompt_string)
    for i in range(len(strings)):
        print(f"Batch lane {i}: {strings[i]}")


def prepare_batch(device: torch.device, batch_loader: BatchLoader,
                  batch_data_preparer: TaskBatchDataPreparer, ids_to_embeddings: EmbedInputFunction,
                  end_of_text_token_id: int, pad_token_id: int,
                  soft_prompt: soft_prompts.SoftPrompt, maximum_prompt_start_indices: int | None = None) -> (
        Tensor, Tensor, Tensor, SoftPromptLossFunction | None, Tensor, Tensor):
    """
    Prepares for a training or test step by requesting samples from the batch loader and getting the data ready.
    :param device: Device to use.
    :param batch_loader: Batch loader to pull samples from.
    :param batch_data_preparer: Data preparer to use for the batch.
    :param ids_to_embeddings: Function that converts token ids to embeddings.
    :param end_of_text_token_id: Token ID of the end of text token.
    :param pad_token_id: Token ID of the pad token.
    :param soft_prompt: Soft prompt to insert into the samples.
    :param maximum_prompt_start_indices: Maximum number of tokens that can occur before the soft prompt.
                                         If None, no maximum is enforced.
    :return: A tuple of (samples, input embeddings, output labels, loss function, soft prompt start indices,
    tokens in batch).
    """
    soft_prompt_parameters, samples, task_metadata = next(batch_loader)
    if soft_prompt_parameters is not None:
        soft_prompt_parameters = soft_prompt_parameters.to(device)
    samples = samples.to(device)
    token_counts = get_token_counts(samples, end_of_text_token_id, pad_token_id)
    maximum_start_indices = torch.clamp(token_counts - 4, min=0, max=maximum_prompt_start_indices)
    soft_prompt_start_indices = (torch.rand(samples.size(0), device=device) * maximum_start_indices).to(
        torch.int64)
    # Note that there's no need to return soft_prompt_parameters because the input_embeddings already contain the
    # soft prompt embedding that was built from the parameters.
    input_embeddings, output_labels, compute_loss = (
        batch_data_preparer.get_batch_data(samples, soft_prompt_start_indices, soft_prompt, soft_prompt_parameters,
                                           task_metadata, ids_to_embeddings, end_of_text_token_id, pad_token_id))
    return samples, input_embeddings, output_labels, compute_loss, soft_prompt_start_indices, torch.sum(token_counts)


def train_and_test_soft_prompt(model, tokenizer,
                               batch_loader: BatchLoader, test_batch_loader: BatchLoader | None,
                               soft_prompt: soft_prompts.SoftPrompt,
                               maximum_prompt_start_indices: int | None,
                               training_step_count: int,
                               batch_data_preparer: TaskBatchDataPreparer,
                               optimizer: torch.optim.Optimizer,
                               accelerator: Accelerator, logger: DataLogger,
                               forward_test_generated_token_count: int = 64,
                               training_loss_logging_interval: int = 1,
                               test_loss_evaluation_interval: int = 32):
    """
    Trains a soft prompt towards some objective defined by samples, output labels, and/or a loss function.

    :param model: The model to use during training.
    :param tokenizer: The tokenizer to use during training.
    :param batch_loader: The batch loader to use during training.
    :param test_batch_loader: The batch loader to use during test loss evaluation and forward testing.
                              Ignored if test_loss_evaluation_interval is not positive.
                              If none, no test loss is evaluated.
    :param soft_prompt: The soft prompt to train.
    :param maximum_prompt_start_indices: The maximum number of tokens that can occur before the soft prompt.
                                         If None, no maximum is enforced.
    :param training_step_count: The number of optimization steps to perform. If gradient accumulation is being used,
                                the number of executed batches will not match the number of optimization steps.
    :param batch_data_preparer: Batch data preparer to use during training.
    :param optimizer: The optimizer to use during training.
    :param accelerator: The accelerator to use during training.
    :param logger: The logger to write to during training.
    :param forward_test_generated_token_count: The number of tokens to generate during forward testing.
    :param training_loss_logging_interval: The interval of training steps at which to log loss.
    :param test_loss_evaluation_interval: The interval of training steps at which to evaluate test loss.
                                          If not positive, no loss tests are performed.
                                          Ignored if test_batch_loader is None.
    """
    ids_to_embeddings = model.get_input_embeddings()
    model, soft_prompt, optimizer = accelerator.prepare(model, soft_prompt, optimizer)
    end_of_text_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    trained_token_count = torch.zeros([1], dtype=torch.int64, device=accelerator.device)
    batch_data_preparer.prepare_preparer(tokenizer, batch_loader.sample_length_in_tokens)
    for training_step_index in range(training_step_count):
        summed_loss = 0.0
        for accumulation_step_index in range(accelerator.gradient_accumulation_steps):
            samples, input_embeddings, output_labels, compute_loss, soft_prompt_start_indices, batch_token_count = (
                prepare_batch(accelerator.device, batch_loader, batch_data_preparer,
                              ids_to_embeddings, end_of_text_token_id, pad_token_id, soft_prompt,
                              maximum_prompt_start_indices))
            trained_token_count += batch_token_count
            summed_loss += run_training_batch(model, optimizer, accelerator, samples, input_embeddings, output_labels,
                                              soft_prompt_start_indices, soft_prompt, compute_loss)

        if logger is not None and training_step_index % training_loss_logging_interval == 0:
            loss = summed_loss / accelerator.gradient_accumulation_steps
            logger.add_scalar('Loss', loss, training_step_index)
            print(f"Completed training step {training_step_index}. Loss: {loss}")
        else:
            print(f"Completed training step {training_step_index}")

        if (test_loss_evaluation_interval > 0 and test_batch_loader is not None and
                training_step_index % test_loss_evaluation_interval == 0):
            # Periodically evaluate test loss.
            samples, input_embeddings, output_labels, compute_loss, soft_prompt_start_indices, batch_token_count = (
                prepare_batch(accelerator.device, test_batch_loader, batch_data_preparer,
                              ids_to_embeddings, end_of_text_token_id, pad_token_id,
                              soft_prompt, maximum_prompt_start_indices))
            test_loss(model, tokenizer, samples, input_embeddings, output_labels, training_step_index,
                      soft_prompt_start_indices, soft_prompt, compute_loss, logger, "[SOFT PROMPT]")
            logger.add_scalar('Trained token count', trained_token_count.to(dtype=torch.float), training_step_index)

    generate_for_forward_testing(test_batch_loader, ids_to_embeddings, model, soft_prompt, tokenizer,
                                 forward_test_generated_token_count,
                                 "[SOFT PROMPT]")
    print(f'completed soft prompt training.')
    logger.add_scalar('Trained token count', trained_token_count.to(dtype=torch.float), training_step_count)
