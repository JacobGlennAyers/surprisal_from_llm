import re
import torch
import torch.nn.functional as F
import pandas as pd
from typing import Generator
from tqdm import tqdm
from collections.abc import Iterable

def error_handler(word_idx, tensor, tokenizer):
    for idx in word_idx:
        if not idx in tensor:
            try:
                sequence = tokenizer.decode(tensor, skip_special_tokens=True)
            except:
                sequence = tokenizer.decode(tensor[0], skip_special_tokens=True)
            raise Exception("An index created by the get_word_ids() cannot be found in the tokenized sequence."
                            " This usually happens with tokenizers that handle space before tokens differently from"
                            " no space before a token. Try passing keep_space=True and make sure sequence initial"
                            " tokens in your data are not preceded by whitespace."
                            "Alternatively, you may attempt to initialize your tokenizer with"
                            " add_prefix_space= True. This may change the outcome of your loss calculations slightly.",
                            f"word indexes: {word_idx}\n",
                            f"index tensor: {tensor}\n",
                            f"encoded_word: {tokenizer.decode(word_idx)}\n"
                            f"encoded_sequence: {sequence}"  # {tokenizer.decode(tensor, skip_special_tokens=True)}\n"
                            f"Another common reason for this error can be whitespaces at the beginning of a sequence")


def get_word_ids_pre_tokenized(sequence: [str], tokenizer, keep_space: bool = False) -> (dict, int):
    if keep_space:
        s = ' '
    else:
        s = ''
    # because tokenizers tend to automatically assume a space before a word in the `is_split_into_words` mode
    # we add spaces in front of all words with the keep_space flag
    sequence_list = [i if idx == 0 else s + i for idx, i in enumerate(sequence)]
    split_sequence = {i: (x, tokenizer.encode(x, add_special_tokens=False))
                      for i, x in enumerate(sequence_list)}
    return split_sequence


def get_word_ids(sequence: str, tokenizer, split_pattern: str = None,
                 keep_space: bool = False) -> (dict, int):
    """'
    splits a string based on a pattern and returns a dictionary where the split token index is the key,
    and the value is a tuple between the string and a list indices
    """
    if keep_space:
        s = ' '
    else:
        s = ''
    if split_pattern:
        if split_pattern == 'tokenizer':
            # first we get the encoding object from the tokenizer
            encoded = tokenizer(sequence)
            try:
                encoded.word_ids()
            except:
                raise Exception('"tokenizer" splitting strategy is currently only implemented for FastTokenizers.'
                                ' The current tokenizer is not a fast tokenizer, please use default or custom splitting'
                                ' instead. A fast tokenizer will be automatically selected if available')
            split_sequence = {}
            # we remove the None values from the word_ids list, this is more verbose than it has to be for readability
            word_ids = [i for i in encoded.word_ids() if i is not None]
            # we leverage the fact that the word ids are organized like a range over the length of the input words
            for i in range(word_ids[-1] + 1):
                # we grab the start and end indexes of each word in the input_ids list
                start, end = encoded.word_to_tokens(i)
                # we decode the relevant indexes to get the word and grab the indexed input_ids
                split_sequence[i] = (tokenizer.decode(encoded.input_ids[start:end]), encoded.input_ids[start:end])
        elif split_pattern == 'token':
            encoded = tokenizer(sequence, add_special_tokens=False)
            words = tokenizer.tokenize(sequence)
            split_sequence = {i: (words[i], [encoded.input_ids[i]]) for i in range(len(words))}
        elif type(split_pattern) is str and 'regex' not in split_pattern:
            if s in split_pattern:
                sequence_list = [i if idx == 0 else s + i for idx, i in enumerate(sequence.split(split_pattern))]
            else:
                sequence_list = [i for _, i in enumerate(sequence.split(split_pattern))]

            split_sequence = {i: (x, tokenizer.encode(x, add_special_tokens=False))
                              for i, x in enumerate(sequence_list)}

        else:
            # we remove the "regex" flag at the beginning of the split pattern
            re_pattern = fr'{split_pattern[5:]}'
            if s in split_pattern:
                sequence_list = [i if idx == 0 else s + i for idx, i in enumerate(re.split(re_pattern, sequence))]
                split_sequence = {i: (x, tokenizer.encode(x, add_special_tokens=False))
                                  for i, x in enumerate(sequence_list)}
            else:
                split_sequence = {i: (x, tokenizer.encode(x, add_special_tokens=False))
                                  for i, x in enumerate(re.split(re_pattern, sequence))}
    else:
        sequence_list = [i if idx == 0 else s + i for idx, i in enumerate(sequence.split())]
        split_sequence = {i: (x, tokenizer.encode(x, add_special_tokens=False))
                          for i, x in enumerate(sequence_list)}
    return split_sequence


def get_word_ids_batch(sequence_list: [str], tokenizer, split_pattern: str or [str] = None,
                       keep_space: bool = False, pre_tokenized=False) -> (dict, int):
    """'
    splits a string based on a pattern and returns a dictionary where the split token index is the key,
    and the value is a tuple between the string and a list indices
    """
    batch_list = []
    for sequence in sequence_list:
        if pre_tokenized:
            split_sequence = get_word_ids_pre_tokenized(sequence, tokenizer, keep_space=keep_space)
        else:
            split_sequence = get_word_ids(sequence, tokenizer, split_pattern, keep_space=keep_space)
        batch_list.append(split_sequence)
    return batch_list


def get_tag_ids(sequence_ids: dict, tag_sequence: str or iter, label2id: dict) -> (dict, int):
    """'
    grabs the ids for the classification tags for classification based surprisal calculations
    """
    #tag_ids = {}
    if type(tag_sequence) is str:
        tag_sequence = tag_sequence.split()
    tag_list = []
    assert len(tag_sequence) == len(
        sequence_ids.keys()), (f"There are {len(tag_sequence)} tags and {len(sequence_ids.keys())} words"
                               f" in your sequence based on your selected tokenization strategy."
                               f" Please ensure the number of tags and words is the same for every sequence")
    for idx in sequence_ids.keys():
        #tag_ids[idx] = (tag_sequence[idx], [label2id[tag_sequence[idx]] for i in sequence_ids[idx][1]])
        tag_list.extend([label2id[tag_sequence[idx]] for i in sequence_ids[idx][1]])
    return tag_sequence, tag_list

def get_tag_ids_batch(sequence_ids: [dict], tag_sequences: iter, label2id: dict) -> (dict, int):
    """'
    splits a string based on a pattern and returns a dictionary where the split token index is the key,
    and the value is a tuple between the string and a list indices
    """
    batch_list = []
    tag_sequence_batch = []
    for index, sequence in enumerate(sequence_ids):
        tag_sequence, tags = get_tag_ids(sequence,tag_sequences[index],label2id)
        batch_list.append(tags)
        tag_sequence_batch.append(tag_sequence)
    return tag_sequence_batch, batch_list

def encode_sequence(sequence: str, tokenizer, tag_sequence: str or iter, label2id: dict,
                    split_pattern: str or [str] = None, return_tensors='pt',
                    pad_len: int = 0, keep_space: bool = False, no_bos=False, pre_tokenized=False,
                    device='cpu') -> dict:
    """
    Returns encodings for each token (based on split pattern) in context of the entire sequence, recommended
    for MlM
    :param keep_space:
    :param pad_len:
    :param sequence:
    :param tokenizer:
    :param split_pattern:
    :param return_tensors:
    :return:
    """
    tokens_expanded = False
    # some tokenizers do not have a built-in pad_id, in those cases we substitute. The token will be ignored
    # since we will set the attention mask to 0
    if tokenizer.pad_token:
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = 0
    if pre_tokenized:
        sequ_ids = get_word_ids_pre_tokenized(sequence, tokenizer, keep_space=keep_space)
    else:
        sequ_ids = get_word_ids(sequence, tokenizer, split_pattern, keep_space=keep_space)

    tags, tag_list = get_tag_ids(sequ_ids, tag_sequence, label2id)

    # We need to join a pre-tokenized sequence back together here because tokenizers generally differ in how they
    # deal with pre-tokenized initial tokens, they will sometimes encode them as a token with a preceeding whitespace
    # even if the initial token has no preceeding whitespace
    if pre_tokenized:
        encode_sequence = tokenizer(" ".join(sequence), return_tensors=return_tensors)
    else:
        encode_sequence = tokenizer(sequence, return_tensors=return_tensors, is_split_into_words=pre_tokenized)
    pad_tensor = torch.tensor([[pad_id for i in range(pad_len - encode_sequence.input_ids.shape[1])]],
                              dtype=encode_sequence.input_ids.dtype)
    pad_attention = torch.tensor([[0 for i in range(pad_len - encode_sequence.attention_mask.shape[1])]],
                                 dtype=encode_sequence.attention_mask.dtype)
    encode_sequence.input_ids = torch.cat((encode_sequence.input_ids, pad_tensor), dim=1)
    encode_sequence.attention_mask = torch.cat((encode_sequence.attention_mask, pad_attention), dim=1)

    mask_counter = 0

    if encode_sequence.input_ids[0, -1] in tokenizer.all_special_ids:
        tag_list.append(0)
    if encode_sequence.input_ids[0, 0] in tokenizer.all_special_ids:
        tag_list = [0] + tag_list
        # we add a 1 to the mask counter so we dont calculate suprisal over a bos token
        mask_counter += 1

    s_range = range(len(sequ_ids))
    for idx in s_range:
        word, word_idx = sequ_ids[idx]
        #print(word, word_idx)
        # this checks if all the word_idx are actually present in the encoded tensor, check the error message
        # for why this is necessary
        # error_handler(word_idx, encode_sequence.input_ids, tokenizer)
        begin_idx = idx + mask_counter
        end_idx = idx + len(word_idx) + mask_counter
        mask_tuple = (0, begin_idx, end_idx)
        mask_counter += end_idx - begin_idx - 1

        mask = torch.isin(encode_sequence.input_ids[0, 1:], torch.tensor(word_idx))
        sequ_ids[idx] = (sequ_ids[idx][0], sequ_ids[idx][1], tags[idx], mask, mask_tuple)

    return {'labels': torch.tensor([tag_list]).to(device),
            "input_ids": encode_sequence.input_ids[:, :].to(device),
            "words": sequ_ids,
            "attention_mask": encode_sequence.attention_mask[:, :].to(device),
            'no_bos': no_bos}


def encode_batch(sequence: [str], tag_sequences: iter, label2id: dict, tokenizer, split_pattern: str or [str] = None, return_tensors='pt',
                 keep_space: bool = False, no_bos=False, padding=True, device='cpu', pre_tokenized=False) -> dict:
    """
    Returns encodings for each token (based on split pattern) in context of the entire sequence, recommended
    for MlM
    :param tag_sequences:
    :param device:
    :param bos_id:
    :param keep_space:
    :param padding:
    :param sequence:
    :param tokenizer:
    :param split_pattern:
    :param return_tensors:
    :return:
    """
    # some tokenizers do not have a built-in pad_id, in those cases we substitute. The token will be ignored
    # since we will set the attention mask to 0
    tokens_expanded = False
    if not tokenizer.pad_token:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    sequence_ids = get_word_ids_batch(sequence, tokenizer, split_pattern, keep_space=keep_space,
                                      pre_tokenized=pre_tokenized)

    encode_sequence = tokenizer(sequence, return_tensors=return_tensors, padding=padding,
                                is_split_into_words=pre_tokenized)

    tags, tag_batch = get_tag_ids_batch(sequence_ids, tag_sequences, label2id)

    if encode_sequence.input_ids[0, -1] in tokenizer.all_special_ids:
        eos_present = True
    else:
        eos_present = False
    if encode_sequence.input_ids[0, 0] in tokenizer.all_special_ids:
        bos_present = True
    else:
        bos_present= False

    for index, sequ in enumerate(sequence_ids):
        # keeps track by how much we've incremented the mask indices
        # mask_counter = 0
        sequ_ids = sequ
        mask_counter = 0
        if bos_present:
            mask_counter +=1
            tag_batch[index] = [0] + tag_batch[index]
        if eos_present:
            tag_batch[index].append(0)
        s_range = range(len(sequ_ids))
        for idx in s_range:
            word, word_idx = sequ_ids[idx]
            begin_idx = idx + mask_counter
            end_idx = idx + len(word_idx) + mask_counter
            mask_tuple = (index, begin_idx, end_idx)
            # if there were more than 1 token corresponding to the word, we increment the mask_counter
            mask_counter += end_idx - begin_idx - 1
            # we save this index_mask so we can use it later,
            # we create a new tuple that now saves the word, the word index from the tokenizer, and the mask
            sequence_ids[index][idx] = (sequence_ids[index][idx][0], sequence_ids[index][idx][1], tags[index][idx], mask_tuple)
    return {'labels': torch.tensor(tag_batch).to(device),
            "input_ids": encode_sequence.input_ids[:, :].to(device),
            "words": sequence_ids,
            "attention_mask": encode_sequence.attention_mask[:, :].to(device),
            'no_bos': no_bos}


def get_surprisal(encoded_sequence: dict, model, entropy=False, logbase='2') -> torch.tensor:
    """
    This function calculates surprisal for a huggingface transformer model over a batch of input.
    The input has to follow the schema produced by encode_batch() or encode_sequence(). It must contain
    input_ids, an attention_mask, and labels. Labels are the input_ids shifted to the right by one index
    :param entropy: flag for calculating shannon entropy
    :param encoded_sequence: {'input_ids': tensor, 'attention_mask': tensor, 'labels': tensor}
    :param model: huggingface transformer
    :return: tensor containing surprisal values of dim [N, S] where N is the batch length and S
    is the sequence length
    """
    if logbase == 'e':
        log = torch.log
    elif logbase == '10':
        log = torch.log10
    else:
        log = torch.log2
    model.eval()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=-1)
        # first we calculate a forward pass over the batch
        output = model(input_ids=encoded_sequence['input_ids'], attention_mask=encoded_sequence['attention_mask'])
        # then we calculate the negative log probabilities over the output, this gives us the nll for all
        output_probs = softmax(output.logits)
        nll = -1 * log(output_probs)
        if entropy:
            entropies = output_probs * nll
            entropies = torch.sum(entropies, dim=-1)
        # in order to find the nll values for the actual tokens in the string, we use torch.gather
        # to find the values at N S V in the output, where N=batch_position, S=sequence_position, V=vocab_index
        # we need to .unsqueeze() the labels so the dimensions correspond to the 3 dimensions of the output
        surprisal = torch.gather(nll, -1, encoded_sequence['labels'].unsqueeze(-1)).squeeze(-1)
    if entropy:
        return surprisal, entropies
    else:
        return surprisal


def return_surprisals(sequence: list or str, tag_sequence: iter or str, model, tokenizer, padding=True, device=None,
                      wordmode=True,
                      sum=True, pre_tokenized: bool = False, no_bos=False, entropy=False, logbase='2', **kwargs):
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model.device.type != device:
        model.to(device)
    label2id = model.config.label2id
    if pre_tokenized:
        encodes = encode_batch(sequence, tag_sequences=tag_sequence, label2id=label2id, tokenizer=tokenizer,
                               device=device, padding=padding,
                               pre_tokenized=pre_tokenized,
                               no_bos=no_bos, **kwargs)
    elif type(sequence) is list:
        encodes = encode_batch(sequence, tag_sequences=tag_sequence, label2id=label2id, tokenizer=tokenizer,
                               device=device, padding=padding, no_bos=no_bos, **kwargs)
    elif type(sequence) is str:
        encodes = encode_sequence(sequence, tag_sequence=tag_sequence, label2id=label2id, tokenizer=tokenizer,
                                  device=device, no_bos=no_bos, **kwargs)
    else:
        raise ValueError('The passed sequence has to be of type list(str) or type str')

    if entropy:
        surp, entropies = get_surprisal(encodes, model, entropy=entropy, logbase=logbase)
    else:
        surp = get_surprisal(encodes, model, logbase=logbase)

    # now we have a surprisal value for every token in the input, but we do not know how those tokens
    # correspond to the words in the sequence, we use the mask_tuple, to grab the correct values from their
    # indices
    if wordmode:
        for index, sequence in enumerate(encodes['words']):
            keys = sorted(sequence.keys())
            for key in keys:
                word = sequence[key][0]
                mask_tuple = sequence[key][-1]
                tag = sequence[key][2]
                if sum:
                    values = torch.sum(surp[mask_tuple[0], mask_tuple[1]:mask_tuple[2]]).item()
                    if entropy:
                        entrop = torch.sum(entropies[mask_tuple[0], mask_tuple[1]:mask_tuple[2]]).item()
                else:
                    # if we do not sum, we commit the tensor to CPU memory, to free up CUDA resources
                    values = surp[mask_tuple[0], mask_tuple[1]:mask_tuple[2]].to('cpu')
                # sequence[key] = (word, values, mask_tuple)
                if entropy:
                    sequence[key] = (word, tag, values, entrop, mask_tuple)
                else:
                    sequence[key] = (word, tag, values, mask_tuple)
            if no_bos:
                sequence[0] = (sequence[0][0], sequence[0][2], 0, 0, 0) if entropy \
                    else (sequence[0][0], sequence[0][2], 0, 0)

    return surp.to('cpu'), encodes['words']


def batch_process(sequence: list or str, tag_sequence: iter or str, model, tokenizer, padding=True,
                  device=None, wordmode=True,
                  sum=True, batch_size=64, no_bos=False, entropy=False, logbase='2', **kwargs):
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model.device.type != device:
        model.to(device)

    for i, _ in enumerate(tqdm(range(0, len(sequence), batch_size), desc='generating surprisal values')):
        start = i * batch_size
        end = start + batch_size
        if end < len(sequence):
            yield return_surprisals(sequence[start:start + batch_size], model=model, tokenizer=tokenizer,
                                    tag_sequence=tag_sequence,
                                    device=device,
                                    padding=padding,
                                    wordmode=wordmode, sum=sum, no_bos=no_bos, entropy=entropy,
                                    logbase=logbase, **kwargs)
        else:
            yield return_surprisals(sequence[start:], model=model, tokenizer=tokenizer,
                                    tag_sequence=tag_sequence,
                                    device=device, padding=padding,
                                    wordmode=wordmode, sum=sum, no_bos=no_bos, entropy=entropy,
                                    logbase=logbase, **kwargs)


def surprisals(sequence: Iterable[str] or Iterable[Iterable], tag_sequence: iter or str, indices: Iterable[any],
               model, tokenizer, padding=True,
               device=None,
               pre_tokenized=False, no_bos=False, entropy=False, logbase='2',
               sum=True, batch_size=64, **kwargs) -> Generator[torch.tensor, dict, Iterable[any]]:
    """
    This function is used to directly pass an iterable object and get surprisal values in return. Intended for import
    and not for calling with the CLI interface
    :param indices:
    :param pre_tokenized:
    :param sequence:
    :param model:
    :param tokenizer:
    :param padding:
    :param device:
    :param sum:
    :param batch_size:
    :param kwargs:
    """
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model.device.type != device:
        model.to(device)

    if isinstance(sequence[0], Iterable) and not isinstance(sequence[0], str):
        if not pre_tokenized:
            print("Detected a sequence of Iterable[Iterable]. Since you did not use the 'pre_tokenized' flag, the "
                  "function will interpret this a s pre_batched sequence. If your input is an Iterable of pre-tokenized"
                  "words, then please enable the 'pre_tokenized=True' flag.")
            for idx, i in enumerate(tqdm(sequence)):
                yield return_surprisals(i, model, tokenizer, device=device, tag_sequence=tag_sequence,
                                        padding=padding, no_bos=no_bos, entropy=entropy, logbase=logbase,
                                        sum=sum, **kwargs), indices[idx] if indices else return_surprisals(
                    i, model, tokenizer, device=device, tag_sequence=tag_sequence,
                    padding=padding, no_bos=no_bos, entropy=entropy, logbase=logbase,
                    sum=sum, **kwargs)
        if pre_tokenized:
            for i, _ in enumerate(tqdm(range(0, len(sequence), batch_size), desc='generating surprisal values')):
                start = i * batch_size
                end = start + batch_size
                if end < len(sequence):
                    yield return_surprisals(sequence[start:start + batch_size], model, tokenizer, device=device,
                                            padding=padding, tag_sequence=tag_sequence,
                                            sum=sum, no_bos=no_bos, entropy=entropy, logbase=logbase,
                                            pre_tokenized=pre_tokenized,
                                            **kwargs), indices[
                                                       start:start + batch_size] if indices else return_surprisals(
                        sequence[start:start + batch_size], model, tokenizer, tag_sequence=tag_sequence,
                        device=device, padding=padding, sum=sum, pre_tokenized=pre_tokenized,
                        no_bos=no_bos, entropy=entropy, logbase=logbase, **kwargs)
                else:
                    yield return_surprisals(sequence[start:], model, tokenizer, device=device, padding=padding,
                                            sum=sum, pre_tokenized=pre_tokenized, no_bos=no_bos, entropy=entropy,
                                            logbase=logbase, tag_sequence=tag_sequence,
                                            **kwargs), indices[start:] if indices else return_surprisals(
                        sequence[start:], model, tokenizer, device=device, padding=padding, tag_sequence=tag_sequence,
                        pre_tokenized=pre_tokenized,
                        sum=sum, no_bos=no_bos, entropy=entropy, logbase=logbase, **kwargs)
    else:
        for i, _ in enumerate(tqdm(range(0, len(sequence), batch_size), desc='generating surprisal values')):
            start = i * batch_size
            end = start + batch_size
            if end < len(sequence):
                yield return_surprisals(sequence[start:start + batch_size], model, tokenizer, device=device,
                                        tag_sequence=tag_sequence,
                                        padding=padding, no_bos=no_bos, entropy=entropy, logbase=logbase,
                                        sum=sum, **kwargs), indices[
                                                            start:start + batch_size] if indices else return_surprisals(
                    sequence[start:start + batch_size], model, tokenizer, tag_sequence=tag_sequence,
                    device=device, padding=padding, no_bos=no_bos, entropy=entropy, logbase=logbase, sum=sum, **kwargs)
            else:
                yield return_surprisals(sequence[start:], model, tokenizer, device=device, padding=padding,
                                        no_bos=no_bos, entropy=entropy, logbase=logbase, tag_sequence=tag_sequence,
                                        sum=sum, **kwargs), indices[start:] if indices else return_surprisals(
                    sequence[start:], model, tokenizer, device=device, padding=padding,
                    no_bos=no_bos, entropy=entropy, logbase=logbase, tag_sequence=tag_sequence,
                    sum=sum, **kwargs)


def surprisals_single_sequence(sequence: list or str, tag_sequence: str or iter, model, tokenizer, keep_space=False,
                               device=None, wordmode=True, logbase = '2',
                               sum=True, pre_tokenized: bool = False, no_bos=False, entropy=False, **kwargs):
    if len(sequence) == 1 and no_bos:
        return 0.0, {0: (sequence[0], None, None)}
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model.device.type != device:
        model.to(device)
    label2id = model.config.label2id
    if pre_tokenized:
        encodes = encode_sequence(sequence, tokenizer, tag_sequence=tag_sequence, label2id=label2id, device=device,
                                  pre_tokenized=pre_tokenized,
                                  keep_space=keep_space,
                                  no_bos=no_bos, **kwargs)
    elif type(sequence) is str:
        encodes = encode_sequence(sequence, tokenizer, tag_sequence=tag_sequence, label2id=label2id,
                                  device=device, keep_space=keep_space,
                                  no_bos=no_bos, **kwargs)
    else:
        raise ValueError('The passed sequence has to be of type list(str) or type str')

    entropy_values = None
    if entropy:
        surp, entropies = get_surprisal(encodes, model, entropy=entropy, logbase=logbase)
    else:
        surp = get_surprisal(encodes, model, logbase=logbase, entropy=False)
    # now we have a surprisal value for every token in the input, but we do not know how those tokens
    # correspond to the words in the sequence, we use the mask_tuple, to grab the correct values from their
    # indices
    # print(encodes)
    if wordmode:
        # we need to make sure we skip the first word if we dont use bos token
        # so we dont have a falsified representation
        keys = sorted(encodes['words'].keys())[1:] if no_bos else sorted(encodes['words'].keys())
        for key in keys:
            word = encodes['words'][key][0]
            mask_tuple = encodes['words'][key][-1]
            if sum:
                values = torch.sum(surp[mask_tuple[0], mask_tuple[1]:mask_tuple[2]]).item()
                if entropy:
                    entropy_values = torch.sum(entropies[mask_tuple[0], mask_tuple[1]:mask_tuple[2]]).item()
            else:
                # if we do not sum, we commit the tensor to CPU memory, to free up CUDA resources
                values = surp[mask_tuple[0], mask_tuple[1]:mask_tuple[2]].to('cpu')
            encodes['words'][key] = (word, values, mask_tuple, entropy_values)
        if no_bos:
            encodes['words'][0] = (encodes['words'][0][0], 0, 0, 0)

    return surp.to('cpu'), encodes['words']
