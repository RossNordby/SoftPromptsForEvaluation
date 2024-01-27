import gc
from numbers import Number
from typing import Callable, TypeAlias

from accelerate import Accelerator
from torch import Tensor
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import torch

from soft_prompting.prompted_embeddings_builder import build_prompted_embeddings
from soft_prompting import soft_prompts, TaskBatchDataPreparer, SoftPromptLossFunction, EmbedInputFunction
from soft_prompting.batch_loader import BatchLoader
from soft_prompting.data_logger import DataLogger
from soft_prompting.training_callbacks import TrainingCallbacks
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
    Evaluates the model and runs an optimization step using the given labels and default loss function,
    or a given loss function if provided.
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


def test_loss_for_batch(model, tokenizer,
                        test_batch_loader, batch_data_preparer, ids_to_embeddings, end_of_text_token_id, pad_token_id,
                        maximum_prompt_start_indices: int | None,
                        accumulator_step_count: int,
                        training_step_index: int,
                        soft_prompt: soft_prompts.SoftPrompt,
                        test_loss_log_title: str | None = None,
                        logger: DataLogger | None = None,
                        soft_prompt_string: str | None = None,
                        print_diagnostic_messages: bool = True):
    """
    Evaluates the loss of the model and prompt without optimizing. Prints diagnostics.
    :param model: Model to train.
    :param tokenizer: Tokenizer used to decode samples.
    :param accumulator_step_count: Number of accumulation steps.
    :param training_step_index: Index of the training step.
    :param soft_prompt: Soft prompt being trained.
    Parameters are (logits, soft_prompt_start_indices).
    :param test_loss_log_title: Title to use when logging the test loss. If None, the logger must None, and if not None,
    the logger must not be None.
    :param logger: Logger to log loss to.
    :param soft_prompt_string: String to insert in the debug outputs to mark the start of the soft prompt, if any.
    If None, no string is inserted.
    :param print_diagnostic_messages: Whether to print diagnostic messages.
    """

    if (logger is None) is not (test_loss_log_title is None):
        raise ValueError("Either both logger and test_loss_log_title must be None, or neither can be None.")

    summed_loss = torch.zeros([1], dtype=torch.float, device=model.device)
    with torch.no_grad():
        for i in range(accumulator_step_count):
            samples, input_embeddings, output_labels, compute_loss, soft_prompt_start_indices, batch_token_count = (
                prepare_batch(model.device, test_batch_loader, batch_data_preparer,
                              ids_to_embeddings, end_of_text_token_id, pad_token_id,
                              soft_prompt, maximum_prompt_start_indices))

            if compute_loss is None:
                # If no custom loss was specified, we'll use the default model loss, so we need to provide labels.
                model_outputs = model.forward(inputs_embeds=input_embeddings, labels=output_labels)
                loss_to_accumulate = model_outputs.loss
            else:
                model_outputs = model.forward(inputs_embeds=input_embeddings)
                # Labels weren't provided. There must be a loss function defined.
                # (logits, labels, soft prompt, soft prompt start indices) -> loss.
                loss_to_accumulate = compute_loss(model_outputs.logits, output_labels, soft_prompt,
                                                  soft_prompt_start_indices)
            if print_diagnostic_messages and i == 0:
                # Bit janky, but it's debug code.
                print(f'Prompt inputs for {training_step_index}')
                print(input_embeddings[0,
                      soft_prompt_start_indices[0].item():soft_prompt_start_indices[
                                                              0].item() + soft_prompt.soft_prompt_token_count, :8])
                debug_input_samples = samples[0]
                debug_model_output = model_outputs.logits[0]
                debug_soft_prompt_start_index = soft_prompt_start_indices[0]
            summed_loss += loss_to_accumulate
    loss = summed_loss / accumulator_step_count
    if logger is not None:
        logger.add_scalar(test_loss_log_title, loss, training_step_index)

    if print_diagnostic_messages:
        print(f"Completed test loss for {training_step_index}, loss: {loss}")
        next_token_ids = sample_token_from_logits(debug_model_output).squeeze()

        print(f'INPUTS, INSERTED SOFT PROMPT AT {debug_soft_prompt_start_index}')
        print(tokenizer.decode(debug_input_samples))
        print()
        print(f'OUTPUTS:')
        # Pull out the soft prompt's predictions:
        before_tokens = next_token_ids[:debug_soft_prompt_start_index]
        after_tokens = next_token_ids[debug_soft_prompt_start_index + soft_prompt.soft_prompt_token_count:]
        before = tokenizer.decode(before_tokens)
        after = tokenizer.decode(after_tokens)
        print(f'{before}{soft_prompt_string}{after}')
        print()
        print("OUTPUTS FOR SOFT PROMPT TOKENS:")
        print(tokenizer.decode(next_token_ids[debug_soft_prompt_start_index:
                                              debug_soft_prompt_start_index + soft_prompt.soft_prompt_token_count]))
        print()
        print("_____")
        print()
    return loss


def generate(prompt_ids: Tensor, model, soft_prompt: soft_prompts.SoftPrompt,
             soft_prompt_parameters: Tensor | None,
             soft_prompt_start_indices: Tensor | None, tokenizer,
             batch_size: int, generated_token_count: int) -> Tensor:
    with torch.no_grad():
        token_counts = get_token_counts(prompt_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
        if soft_prompt_start_indices is None:
            soft_prompt_start_indices = token_counts
        else:
            soft_prompt_start_indices = torch.clamp(soft_prompt_start_indices, max=token_counts)

        raw_embeddings = model.get_input_embeddings()(prompt_ids)

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


def generate_from_prompts(prompts: list[str], soft_prompt_parameters: None | Tensor | list[tuple[Number, ...]],
                          soft_prompt_start_indices: None | Tensor | list[int] | int,
                          soft_prompt: soft_prompts.SoftPrompt, model, tokenizer,
                          batch_size: int, generated_token_count: int) -> (Tensor, Tensor):
    """
    Generates tokens from a list of prompts.
    :param prompts: Prompts to generate from.
    :param soft_prompt_parameters: Soft prompt parameters to use. If None, no parameters are used.
    :param soft_prompt_start_indices: Indices to insert the soft prompt at. If None, the soft prompt is inserted at the
                                      end of the prompt. If an int, the same index is used for all prompts.
                                      Any indices beyond the end of the prompt are clamped to the end of the prompt.
    :param soft_prompt: Soft prompt to insert.
    :param model: Model to generate with.
    :param tokenizer: Tokenizer to use.
    :param batch_size: Batch size to use.
    :param generated_token_count: Number of tokens to generate.
    :return: A tuple of (input prompt token ids, generated token ids).
    """
    # Convert non-none nontensor inputs to tensors.
    if soft_prompt_parameters is not None and not isinstance(soft_prompt_parameters, Tensor):
        soft_prompt_parameters = torch.tensor(soft_prompt_parameters)
    if (soft_prompt_start_indices is not None and
            not isinstance(soft_prompt_start_indices, Tensor) and
            not isinstance(soft_prompt_start_indices, int)):
        soft_prompt_start_indices = torch.tensor(soft_prompt_start_indices)
    elif isinstance(soft_prompt_start_indices, int):
        soft_prompt_start_indices = torch.tensor([soft_prompt_start_indices] * len(prompts))

    prompt_ids = tokenizer(prompts, return_tensors='pt', padding=True).input_ids
    # Make sure everything is on the same device.
    if soft_prompt_start_indices is not None:
        soft_prompt_start_indices = soft_prompt_start_indices.to(model.device)
    if soft_prompt_parameters is not None:
        soft_prompt_parameters = soft_prompt_parameters.to(model.device)
    prompt_ids = prompt_ids.to(model.device)
    return prompt_ids, generate(prompt_ids, model, soft_prompt, soft_prompt_parameters, soft_prompt_start_indices,
                                tokenizer,
                                batch_size, generated_token_count)


def generate_from_batch_loader(batch_loader: BatchLoader, model,
                               soft_prompt: soft_prompts.SoftPrompt, tokenizer,
                               generated_token_count: int) -> (Tensor, Tensor):
    """
    Generates tokens using prompts pulled from a batch loader.
    :param batch_loader: Batch loader to pull prompts from.
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
            generate(samples, model, soft_prompt, soft_prompt_parameters, None, tokenizer, samples.size(0),
                     generated_token_count))


def create_strings_from_prompted_generation(prompt_ids: Tensor, generated_ids: Tensor, tokenizer,
                                            soft_prompt_start_indices: Tensor | None = None,
                                            soft_prompt_string: str | None = None) -> list[str]:
    """
    Creates strings for each lane in a batch from prompt ids and generated ids, optionally including a marker for
    where the soft prompt was inserted.
    :param prompt_ids: Prompt ids with dimensions [batch_size, token_count], Lanes may be right-padded.
    :param generated_ids: Generated ids with dimensions [batch_size, token_count].
    :param tokenizer: Tokenizer used to decode ids.
    :param soft_prompt_start_indices: Indices of the soft prompt in the prompt ids. If None, the soft prompt was
    placed at the end of the prompt.
    :param soft_prompt_string: String to insert to mark the start of the soft prompt, if any. If None, no string is
    inserted.
    :return: A list of strings for the batch.
    """
    token_counts = get_token_counts(prompt_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
    strings = []
    if isinstance(soft_prompt_start_indices, int):
        soft_prompt_start_indices = torch.tensor([soft_prompt_start_indices] * prompt_ids.size(0))
    for i in range(prompt_ids.size(0)):
        if (soft_prompt_start_indices is not None and soft_prompt_start_indices[i] < token_counts[i] and
                soft_prompt_string is not None):
            pre_soft_prompt_string = tokenizer.decode(prompt_ids[i, :soft_prompt_start_indices[i]])
            post_soft_prompt_string = tokenizer.decode(prompt_ids[i, soft_prompt_start_indices[i]:token_counts[i]])
            prompt_string = pre_soft_prompt_string + soft_prompt_string + post_soft_prompt_string
        else:
            prompt_string = tokenizer.decode(prompt_ids[i, :token_counts[i]])
            if soft_prompt_string is not None:
                prompt_string += soft_prompt_string
        generated_string = tokenizer.decode(generated_ids[i])
        strings.append(prompt_string + generated_string)
    return strings


def generate_for_forward_testing(batch_loader: BatchLoader, model,
                                 soft_prompt: soft_prompts.SoftPrompt, tokenizer, generated_token_count: int,
                                 prompt_string: str | None = None):
    """
    Generates tokens using prompts pulled from a batch loader and prints them.
    :param batch_loader: Batch loader to pull prompts from.
    :param model: Model to generate with.
    :param soft_prompt: Soft prompt to insert after the prompt.
    :param tokenizer: Tokenizer to use.
    :param generated_token_count: Number of tokens to generate.
    :param prompt_string: String to insert to mark the start of the soft prompt, if any. If None, no string is
    inserted.
    """
    input_ids, output_ids = generate_from_batch_loader(batch_loader, model, soft_prompt, tokenizer,
                                                       generated_token_count)
    strings = create_strings_from_prompted_generation(input_ids, output_ids, tokenizer, None, prompt_string)
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


def train_and_test_soft_prompt(model, model_name: str, tokenizer,
                               batch_loader: BatchLoader, test_batch_loader: BatchLoader | None,
                               soft_prompt: soft_prompts.SoftPrompt,
                               maximum_prompt_start_indices: int | None,
                               training_step_count: int,
                               batch_data_preparer: TaskBatchDataPreparer,
                               optimizer: torch.optim.Optimizer,
                               accelerator: Accelerator, logger: DataLogger,
                               forward_test_generated_token_count: int = 64,
                               training_loss_logging_interval: int = 1,
                               test_loss_evaluation_interval: int = 32,
                               final_test_loss_evaluation_step_count: int = 64,
                               training_callbacks: TrainingCallbacks | None = None):
    """
    Trains a soft prompt towards some objective defined by samples, output labels, and/or a loss function.

    :param model: The model to use during training.
    :param model_name: The name of the model to use during training.
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
    :param final_test_loss_evaluation_step_count: The number of steps to use when evaluating test loss at the end of
                                                  training. Ignored if test_batch_loader is None.
    :param training_callbacks: Callbacks to run during training.
    """
    ids_to_embeddings = model.get_input_embeddings()
    model, soft_prompt, optimizer = accelerator.prepare(model, soft_prompt, optimizer)
    end_of_text_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    trained_token_count = torch.zeros([1], dtype=torch.int64, device=accelerator.device)
    batch_data_preparer.prepare_preparer(tokenizer, batch_loader.sample_length_in_tokens)
    for training_step_index in range(training_step_count):
        summed_loss = torch.zeros([1], dtype=torch.float, device=accelerator.device)
        for accumulation_step_index in range(accelerator.gradient_accumulation_steps):
            samples, input_embeddings, output_labels, compute_loss, soft_prompt_start_indices, batch_token_count = (
                prepare_batch(accelerator.device, batch_loader, batch_data_preparer,
                              ids_to_embeddings, end_of_text_token_id, pad_token_id, soft_prompt,
                              maximum_prompt_start_indices))
            trained_token_count += batch_token_count
            summed_loss += run_training_batch(model, optimizer, accelerator, samples, input_embeddings, output_labels,
                                              soft_prompt_start_indices, soft_prompt, compute_loss)

        if model.device.type == 'cuda':
            # If we're running on a GPU, keep an eye on the cache to avoid runaway fragmentation.
            # Observed this on some of the more variable training runs.
            total_memory = torch.cuda.get_device_properties(model.device).total_memory
            allocated_memory = torch.cuda.memory_allocated(model.device)
            cached_memory = torch.cuda.memory_reserved(model.device)
            if cached_memory > total_memory * 0.15 and cached_memory + allocated_memory >= total_memory:
                torch.cuda.empty_cache()
                gc.collect()

        if logger is not None and training_step_index % training_loss_logging_interval == 0:
            loss = summed_loss / accelerator.gradient_accumulation_steps
            logger.add_scalar('Loss', loss, training_step_index)
            print(f"Completed training step {training_step_index}. Loss: {loss.item()}")
        else:
            print(f"Completed training step {training_step_index}")

        if (test_loss_evaluation_interval > 0 and test_batch_loader is not None and
                training_step_index % test_loss_evaluation_interval == 0):
            # Periodically evaluate test loss.
            test_loss_for_batch(model, tokenizer, test_batch_loader,
                                batch_data_preparer, ids_to_embeddings, end_of_text_token_id, pad_token_id,
                                maximum_prompt_start_indices,
                                accelerator.gradient_accumulation_steps, training_step_index,
                                soft_prompt, "Test Loss", logger,
                                "[SOFT PROMPT]")
            logger.add_scalar('Trained token count', trained_token_count.to(dtype=torch.float), training_step_index)

    generate_for_forward_testing(test_batch_loader, model, soft_prompt, tokenizer, forward_test_generated_token_count,
                                 "[SOFT PROMPT]")
    logger.add_scalar('Trained token count', trained_token_count.to(dtype=torch.float), training_step_count)
    print(f'completed soft prompt training.')

    if final_test_loss_evaluation_step_count > 0 and test_batch_loader is not None:
        summed_loss = torch.zeros([1], dtype=torch.float, device=accelerator.device)
        for i in range(final_test_loss_evaluation_step_count):
            summed_loss += test_loss_for_batch(model, tokenizer, test_batch_loader,
                                               batch_data_preparer, ids_to_embeddings, end_of_text_token_id,
                                               pad_token_id, maximum_prompt_start_indices,
                                               accelerator.gradient_accumulation_steps, i,
                                               soft_prompt, "Final Test Loss", logger,
                                               "[SOFT PROMPT]")
        loss = summed_loss / final_test_loss_evaluation_step_count
        # Also janky, but we're very low on time and just need a way to persist the final test loss.
        logger.add_scalar('Final Test Loss Average', loss, 0)
    if training_callbacks is not None:
        training_callbacks.training_complete(model_name, model, tokenizer, batch_loader.sample_length_in_tokens,
                                             batch_loader.batch_size,
                                             accelerator.gradient_accumulation_steps,
                                             soft_prompt, training_step_count,
                                             optimizer.param_groups[0]['lr'],
                                             optimizer.param_groups[0]['weight_decay'])
