import torch
import numpy as np
import torchtext
from typing import List
from ranking_scores import compute_cosine_similarity
from utils import clean_tokens, check_pos_bert, check_pos_word

def run_batched_suggestion_generation(dataloader, model, tokenizer, num_suggestions, device):
    """
    Runs suggestion generation in batches using a PyTorch DataLoader.
    """
    all_suggestions = []
    
    print("Batched suggestion generation running...")

    batch_count = 1

    for batch in dataloader:
        batch_count += 1
        #print("batch")
        #print(batch)
        masked_inputs_batch = batch["masked_inputs_batch"]  # Batched tokenized inputs
        #replaced_words_batch = batch["words_found_batch"]  # Batched replaced words
        replaced_tokens_batch = batch["tokens_found_batch"] # Batched replaced tokens

        # Generate suggestions for the entire batch
        batch_suggestions = suggestion_generator(
            model, tokenizer, masked_inputs_batch, replaced_tokens_batch, num_suggestions, device
        )
        all_suggestions.extend(batch_suggestions)

    return all_suggestions


def suggestion_generator(model, tokenizer, masked_inputs, replaced_tokens_batch, num_suggestions, device):
    """
    Generates suggestions for masked words using a BERT-based masked language model in batch mode.

    Args:
        model: The BERT-based model for masked language modeling.
        tokenizer: The tokenizer used for tokenizing the text.
        masked_inputs (dict): The tokenized input with [MASK] tokens, batched.
        #replaced_words_batch (list of lists): A batch of lists containing words that were replaced with [MASK] in each sentence.
        replaced_tokens_batch (list of lists): A batch of lists containing tokens that were replaced with [MASK] in each sentence.
        num_suggestions (int): The number of suggestions to generate for each masked word.
        device (str): The device to run the model on.

    Returns:
        batch_suggestions (list of dicts): A list of dictionaries, each mapping replaced tokens to lists of suggestions.
    """

    # Move model and input data to the specified device
    model.to(device)
    masked_inputs = {key: val.to(device) for key, val in masked_inputs.items()}

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**masked_inputs)
    logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

    batch_suggestions = []

    print("replaced_tokens_batch")
    print(replaced_tokens_batch)

    # Process each sentence in the batch
    for batch_idx, replaced_tokens in enumerate(replaced_tokens_batch):
        suggestions = {}

        # Identify masked token positions
        masked_indices = (masked_inputs["input_ids"][batch_idx] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

        # For each masked token, get the top-k suggestions
        for idx, word in zip(masked_indices, replaced_tokens):
            token_logits = logits[batch_idx, idx]

            # Get top-k predicted token IDs
            top_k_vals, top_k_ids = torch.topk(token_logits, num_suggestions)
            top_k_vals = top_k_vals.tolist()
            top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids.tolist())

            # Store suggestions
            suggestions[word] = [(token, val) for token, val in zip(top_k_tokens, top_k_vals)]

        batch_suggestions.append(suggestions)

    return batch_suggestions

def suggestion_filtering(all_suggestions: List, all_cleaned_texts: List, 
                         all_words_found, 
                         df_subtlex, complex_words, eng_words, glove, 
                         tokenizer, 
                         num_suggestions_filtered):
    
    print("Filtering the suggestions based on pos tags and cos similarity scores...")
    all_filtered_suggestions = []

    is_glove = isinstance(glove, torchtext.vocab.GloVe)
    for suggestion_dictionary, original_text, words_found in zip(all_suggestions, all_cleaned_texts, all_words_found):
        filtered_suggestion_dict = {}
        for key, suggestions in suggestion_dictionary.items():
            cosine_score_list = []
            candidate_list = []
            bert_score_list = []

            key_clean = key[2:] if key.startswith("##") else key
            # Clean the target word tokens
            target_word_wo_occ = key.split("_[")[0] if "_[" in key else key
            occurrence_count = int(key.split("_[")[1][:-1]) if "_[" in key else 1
            target_word_clean = key_clean.split("_[")[0] if "_[" in key_clean else key_clean
            target_word = target_word_wo_occ.split()[0] if len(target_word_wo_occ.split()) > 1 else target_word_wo_occ

            for candidate, bert_score in suggestions:                
                if candidate.startswith("##"):
                    if not(key.startswith("##")):
                        continue
                    else:
                        candidate_clean = candidate[2:]
                        #key_clean = key[2:]
                else:
                    #if key.startswith("##"):
                    #    continue
                    candidate_clean = candidate
                    #key_clean = key

                # Check if the word is not found in corpora or is also a complex word
                is_complex = ((candidate_clean not in eng_words) or (candidate_clean in complex_words) or (candidate_clean == target_word_clean)) 
                if is_complex:
                    continue

                # Check part-of-speech compatibility
                #pos_flag = check_pos_bert(original_text=original_text, target_word=target_word_wo_occ, suggestion=candidate, tokenizer=tokenizer, occurrence_count=occurrence_count)
                pos_flag = check_pos_word(original_text=original_text, target_word=target_word_wo_occ, suggestion=candidate, occurrence_count=occurrence_count)

                if not(pos_flag):
                    continue

                cosine_score = compute_cosine_similarity(target_word_clean, candidate_clean, glove)
                cosine_score_list.append(cosine_score)
                candidate_list.append(candidate)
                bert_score_list.append(bert_score)

            top_n_candidates = [candidate_list[i] for i in np.argsort(cosine_score_list)[::-1]][:num_suggestions_filtered]
            top_n_bert_scores = [bert_score_list[i] for i in np.argsort(cosine_score_list)[::-1]][:num_suggestions_filtered]

            filtered_suggestion_dict[key] = [(candidate, bert_score) for candidate, bert_score in zip(top_n_candidates, top_n_bert_scores)]
        
        all_filtered_suggestions.append(filtered_suggestion_dict)
    
    print("Suggestion filtering completed.")
    
    return all_filtered_suggestions
