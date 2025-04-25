import re
import copy
import language_tool_python
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer, SimilarityFunction 
from sentence_transformers.models import Pooling, Transformer
from nltk.tokenize.treebank import TreebankWordDetokenizer

def batch_true_correct(sentences, words, lang_tool):
    """
    Corrects a batch of sentences efficiently.
    
    Args:
        sentences (list of str): List of sentences to be corrected.
        words (set or list, optional): If provided, only correct mistakes involving these words.
        
    Returns:
        list of str: List of corrected sentences.
    """
    return [true_correct(sentence, words, lang_tool) for sentence in sentences]

def true_correct(text, words, lang_tool):
    matches = lang_tool.check(text)
    corrections = []

    # Extracting mistakes and corrections
    for rules in matches:
        if rules.replacements:
            mistake = text[rules.offset:rules.offset + rules.errorLength]

            # Ensure mistake is within the words list (if provided)
            if words is None or any(mistake.lower() == word.lower() for word in words):
                corrections.append((rules.offset, rules.offset + rules.errorLength, rules.replacements[0]))

    # Sort corrections in reverse order to avoid index shifting issues
    corrections.sort(reverse=True, key=lambda x: x[0])

    # Apply corrections
    for start, end, correction in corrections:
        text = text[:start] + correction + text[end:]

    return text

def ner_checker(nlp_ner, sentence, word):
    # Checks whether the word is part of an entity
    document = nlp_ner(sentence)
    for token in document:
        if token.text.lower() == word.lower():
            return token.ent_iob_ != 'O'  # True if part of an entity
    return False  # Word not found or not an entity

def check_pos_word(original_text, target_word, suggestion, occurrence_count=1):
    POS_MAPPING = {
        "JJ": ["NN", "NNS"],  # Adjective ↔ Noun
        "NN": ["JJ"],          # Noun ↔ Adjective
        "VB": ["NN", "VB"],          # Verb ↔ Noun
    }

    # Tokenize the texts
    original_tokens = nltk.word_tokenize(original_text)

    suggested_text, suggestion_index = place_suggestion_into_sentence_word(original_text, target_word, suggestion, target_occurrence_count=occurrence_count)
    
    suggestion_tokens = nltk.word_tokenize(suggested_text)

    try:
        # Get POS tagging on full sentences
        original_pos_tags = dict(nltk.pos_tag(original_tokens))
        suggested_pos_tags = dict(nltk.pos_tag(suggestion_tokens))
        
        ## Find POS for the specific word
        original_pos = original_pos_tags.get(target_word, None)
        suggested_pos = suggested_pos_tags.get(suggestion, None)

        # Debugging outputs
        #print(f"Original Word: {target_word}, POS: {original_pos}")
        #print(f"Suggested Word: {suggestion}, POS: {suggested_pos}")

        # If POS couldn't be determined, return False
        if not original_pos or not suggested_pos:
            return False

        # Allow verbs to be replaced by other verbs
        if original_pos == suggested_pos or suggested_pos in POS_MAPPING.get(original_pos, []):
            return True
        return False

    except Exception as e:
        print(f"Error in POS Matching: {e}")
        return False


def check_pos_bert(original_text, target_word, suggestion, tokenizer, occurrence_count=1):
    POS_MAPPING = {
        "JJ": ["NN", "NNS"],  # Adjective ↔ Noun
        "NN": ["JJ"],          # Noun ↔ Adjective
        "VB": ["NN", "VB"],          # Verb ↔ Noun
    }

    # Tokenize the texts
    original_tokens = nltk.word_tokenize(original_text)

    original_text_list = original_text.split() #works when operating on cleaned text, may be problematic otherwise, check this!

    observed_target = 0
    for l in range(len(original_text_list)):
        word = original_text_list[l]
        word_tokenize = tokenizer.tokenize(word)
        if target_word in word_tokenize:
            observed_target += 1
            if observed_target == occurrence_count:
                curr_idx = l
                break
    curr_word = original_text_list[curr_idx]
    curr_word_tokenized = tokenizer.tokenize(curr_word)
    for t in range(len(curr_word_tokenized)):
        if curr_word_tokenized[t] == target_word:
            target_token = curr_word_tokenized
            curr_word_tokenized[t] = suggestion
    #if any("##" in token for token in curr_word_tokenized):
    #    return True
    suggested_word = tokenizer.convert_tokens_to_string(curr_word_tokenized)
    original_text_list[curr_idx] = suggested_word
    suggested_text = " ".join(original_text_list)
    
    """
    original_bert_tokens = tokenizer.tokenize(original_text)
    start_idx = 0
    for word_count in range(occurrence_count):
        curr_idx = original_bert_tokens.index(target_word, start_idx)
        start_idx = curr_idx + 1
    original_bert_tokens[curr_idx] = suggestion
    suggested_text = tokenizer.convert_tokens_to_string(original_bert_tokens)
    """
    
    suggestion_tokens = nltk.word_tokenize(suggested_text)

    try:
        # Get POS tagging on full sentences
        original_pos_tags = dict(nltk.pos_tag(original_tokens))
        suggested_pos_tags = dict(nltk.pos_tag(suggestion_tokens))

        for word in original_pos_tags.keys():
            if target_word in tokenizer.tokenize(word) or target_word == word:
                original_pos = original_pos_tags.get(word, None)

        for word in suggested_pos_tags.keys():
            if suggestion in tokenizer.tokenize(word) or suggestion == word:
                suggested_pos = suggested_pos_tags.get(word, None)        
        
        ## Find POS for the specific word
        #original_pos = original_pos_tags.get(target_word, None)
        #suggested_pos = suggested_pos_tags.get(suggestion, None)

        # Debugging outputs
        #print(f"Original Word: {target_word}, POS: {original_pos}")
        #print(f"Suggested Word: {suggestion}, POS: {suggested_pos}")

        # If POS couldn't be determined, return False
        if not original_pos or not suggested_pos:
            return False

        # Allow verbs to be replaced by other verbs
        if original_pos == suggested_pos or suggested_pos in POS_MAPPING.get(original_pos, []):
            return True
        return False

    except Exception as e:
        print(f"Error in POS Matching: {e}")
        return False

def check_pos(original_text, target_word, suggestion, occurrence_count=1):
    POS_MAPPING = {
        "JJ": ["NN", "NNS"],  # Adjective ↔ Noun
        "NN": ["JJ"],          # Noun ↔ Adjective
        "VB": ["NN", "VB"],          # Verb ↔ Noun
    }

    # Tokenize the texts
    original_tokens = nltk.word_tokenize(original_text)
    suggested_text = original_text.replace(target_word, suggestion, occurrence_count)
    suggestion_tokens = nltk.word_tokenize(suggested_text)

    try:
        # Get POS tagging on full sentences
        original_pos_tags = dict(nltk.pos_tag(original_tokens))
        suggested_pos_tags = dict(nltk.pos_tag(suggestion_tokens))

        # Find POS for the specific word
        original_pos = original_pos_tags.get(target_word, None)
        suggested_pos = suggested_pos_tags.get(suggestion, None)

        # Debugging outputs
        #print(f"Original Word: {target_word}, POS: {original_pos}")
        #print(f"Suggested Word: {suggestion}, POS: {suggested_pos}")

        # If POS couldn't be determined, return False
        if not original_pos or not suggested_pos:
            return False

        # Allow verbs to be replaced by other verbs
        if original_pos == suggested_pos or suggested_pos in POS_MAPPING.get(original_pos, []):
            return True
        return False

    except Exception as e:
        print(f"Error in POS Matching: {e}")
        return False

def check_pos_old(original_text, target_word, suggestion, occurrence_count=1):
    POS_MAPPING = {
        "JJ": ["NN", "NNS"],  # Adjective ↔ Noun
        "NN": ["JJ"],          # Noun ↔ Adjective
        ## Refined verb mappings
        #"VB": ["VBP", "VBG"],   # Base form ↔ Present (plural) & Gerund
        #"VBP": ["VB", "VBZ"],   # Present (plural) ↔ Base & 3rd person singular
        #"VBZ": ["VBP", "VB"],   # 3rd person singular ↔ Present (plural) & Base
        #"VBD": ["VBN", "VBD"],  # Past tense ↔ Past participle
        #"VBN": ["VBD", "VBN"],  # Past participle ↔ Past tense
        #"VBG": ["VB", "VBG"],   # Gerund ↔ Base form
        "VB": ["NN"],          # Verb ↔ Noun
    }
    """
    Check if the POS tags of the target word and its simplification are compatible.
    """
    original_tokens = nltk.word_tokenize(original_text)
    suggested_text = original_text.replace(target_word, suggestion, occurrence_count)
    suggestion_tokens = nltk.word_tokenize(suggested_text)

    try:
        ind = original_tokens.index(target_word)
        original_pos = nltk.pos_tag([original_tokens[ind]])[0][1]
        suggested_pos = nltk.pos_tag([suggestion_tokens[ind]])[0][1]
        if target_word == "provides" and suggestion == "allows":
            print("provides, allows")
            print(f"Original POS: {original_pos}, Suggested POS: {suggested_pos}")
            print("original_tokens")
            print(original_tokens)
            print("suggestion_tokens")
            print(suggestion_tokens)
            print("nltk.pos_tag([original_tokens[ind]])")
            print(nltk.pos_tag([original_tokens[ind]]))
            print("nltk.pos_tag([suggestion_tokens[ind]])")
            print(nltk.pos_tag([suggestion_tokens[ind]]))

            # Get POS tagging on full sentences
            original_pos_tags = dict(nltk.pos_tag(original_tokens))
            suggested_pos_tags = dict(nltk.pos_tag(suggestion_tokens))

            # Find POS for the specific word
            original_pos = original_pos_tags.get(target_word, None)
            suggested_pos = suggested_pos_tags.get(suggestion, None)

            # Debugging outputs
            print(f"Original Word: {target_word}, POS: {original_pos}")
            print(f"Suggested Word: {suggestion}, POS: {suggested_pos}")
            
            print("nltk.pos_tag([original_tokens])")
            print(nltk.pos_tag([original_tokens]))
            print("nltk.pos_tag([suggestion_tokens])")
            print(nltk.pos_tag([suggestion_tokens]))

        # Allow flexibility in POS conversion
        if original_pos == suggested_pos or suggested_pos in POS_MAPPING.get(original_pos, []):
            return True
        return False

    except (ValueError, IndexError) as e:
        print(f"Error in POS Matching: {e}")
        return False


def check_pos_strict(original_text, target_word, suggestion):
    """
    Check if the POS tags of the target word and its substitution match in context.
    """
    # Tokenize the original and modified texts
    original_tokens = nltk.word_tokenize(original_text)

    # Ensure we only replace the first occurrence of the target word
    suggested_text = original_text.replace(target_word, suggestion, 1)
    suggestion_tokens = nltk.word_tokenize(suggested_text)

    try:
        # Find the index of the target word in the original sentence
        ind = original_tokens.index(target_word)

        # POS tag both sentences
        original_pos = nltk.pos_tag(original_tokens)
        suggested_pos = nltk.pos_tag(suggestion_tokens)

        # Compare the POS tags at the same index
        return original_pos[ind][1][0] == suggested_pos[ind][1][0]

    except ValueError:
        # Handle cases where the word is not found
        print(f"Word '{target_word}' not found in tokens.")
        return False
    except IndexError:
        # Handle cases where the index is out of range
        print(f"Index error for '{target_word}' with suggestion '{suggestion}'.")
        return False

def sentence_builder(original_text, suggested_tokens, tokenizer=nltk.word_tokenize):
    produced_text_list = nltk.word_tokenize(original_text) #works when operating on cleaned text, may be problematic otherwise, check this!
    for target_word in suggested_tokens.keys():
        for l in range(len(produced_text_list)):
            word = produced_text_list[l]
            if target_word.lower() == word.lower():
                produced_text_list[l] = suggested_tokens[target_word]
                break
    produced_text = TreebankWordDetokenizer().detokenize(produced_text_list)
    return produced_text

def sentence_builder_old(original_text, suggested_tokens, tokenizer):
    produced_text_list = original_text.split() #works when operating on cleaned text, may be problematic otherwise, check this!

    for target_word in suggested_tokens.keys():
        for l in range(len(produced_text_list)):
            word = produced_text_list[l]
            word_tokenize = tokenizer.tokenize(word)
            if target_word in word_tokenize:
                curr_idx = l
                break
        curr_word = produced_text_list[curr_idx]
        curr_word_tokenized = tokenizer.tokenize(curr_word)
        for t in range(len(curr_word_tokenized)):
            if curr_word_tokenized[t] == target_word:
                curr_word_tokenized[t] = suggested_tokens[target_word]
        suggested_word = tokenizer.convert_tokens_to_string(curr_word_tokenized)
        produced_text_list[curr_idx] = suggested_word
        produced_text = " ".join(produced_text_list)
    return produced_text

def sentence_builder_word(original_text, suggested_words):
    produced_text = copy.deepcopy(original_text)
    for key in sorted(list(suggested_words.keys())):
        if "_[" in key:
            target_word = key.split("_[")[0]
            occurence_count = int(key.split("_[")[1][:-1])
        else:
            target_word = key
            occurence_count = 1
        if target_word in original_text:
            # only replaces one instance
            # since the keys are sorted, repeating complex word instances will be successfully distinguished
            produced_text = re.sub(f" {target_word} ", f" {suggested_words[key]} ", produced_text, 1)
    return produced_text

def load_sentence_model(model, device):
    model_name_or_path = model.name_or_path
    max_token_length = 512
    word_embedding_model = Transformer(model_name_or_path, max_seq_length=max_token_length)
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension()) #mean pooling by default
    sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
    #sentence_model.similarity_fn_name = SimilarityFunction.MANHATTAN
    #sentence_model.to(device)
    return sentence_model

def place_suggestion_into_sentence_word(original_text, target_word, suggestion, target_occurrence_count=1):
    original_text_list = original_text.split() #works when operating on cleaned text, may be problematic otherwise, check this!
    observed_target = 0
    for l in range(len(original_text_list)):
        word = original_text_list[l]
        if target_word == word:
            observed_target += 1
            if observed_target == target_occurrence_count:
                curr_idx = l
                break
    original_text_list[curr_idx] = suggestion
    modified_text = " ".join(original_text_list)
    suggestion_index = curr_idx
    return modified_text, suggestion_index

def place_suggestion_into_sentence_token(original_text, target_word, suggestion, tokenizer, target_occurrence_count=1):
    original_text_list = original_text.split() #works when operating on cleaned text, may be problematic otherwise, check this!
    observed_target = 0
    for l in range(len(original_text_list)):
        word = original_text_list[l]
        word_tokenize = tokenizer.tokenize(word)
        if target_word in word_tokenize:
            observed_target += 1
            if observed_target == target_occurrence_count:
                curr_idx = l
                break
    #curr_idx = original_text_list.index(target_word, start_idx)
    #curr_idx = original_text_tokens.index(target_word, start_idx)
    curr_word = original_text_list[curr_idx]
    curr_word_tokenized = tokenizer.tokenize(curr_word)
    for t in range(len(curr_word_tokenized)):
        if curr_word_tokenized[t] == target_word:
            curr_word_tokenized[t] = suggestion
    suggested_word = tokenizer.convert_tokens_to_string(curr_word_tokenized)
    original_text_list[curr_idx] = suggested_word
    modified_text = " ".join(original_text_list)
    suggestion_index = curr_idx
    return modified_text, suggestion_index

def compute_global_stats(global_scores):
    min_global = np.min(global_scores)
    max_global = np.max(global_scores)
    mean_global = np.mean(global_scores)
    std_global = np.std(global_scores)
    return {"min": min_global, "max": max_global, "mean": mean_global, "std": std_global}

def normalize_score(x, score_type, global_stats=None, global_scores=None):
    """
    Normalize scores to [0,1] range with appropriate scaling per score type.
    
    Parameters:
    - x: score value(s) to be normalized.
    - score_type: type of score (determines normalization method).
    - global_stats: dictionary with precomputed min, max, mean, std.

    Returns:
    - Normalized score value(s) in range [0,1].
    """
    
    # Ensure `global_score_array` is a NumPy array
    global_score_array = np.array(global_scores) if global_scores is not None else np.array(x)

    # Compute statistics if not provided
    if global_stats is None:
        min_global = np.min(global_score_array)
        max_global = np.max(global_score_array)
        mean_global = np.mean(global_score_array)
        std_global = np.std(global_score_array)
    else:
        min_global = global_stats.get("min", np.min(global_score_array))
        max_global = global_stats.get("max", np.max(global_score_array))
        mean_global = global_stats.get("mean", np.mean(global_score_array))
        std_global = global_stats.get("std", np.std(global_score_array))

    # Prevent division by zero for std
    std_global = max(std_global, 1e-8)

    # Min-Max Scaling (Broadly distributed scores)
    if score_type in ["bert", "freq", "cos", "lm"]:
        return (x - min_global) / (max_global - min_global + 1e-8)

    # Log Scaling (Highly skewed distribution)
    elif score_type == "len":
        return (np.log1p(x) - np.log1p(min_global)) / (np.log1p(max_global) - np.log1p(min_global))

    ## Z-Score Normalization (Scores close to [0,1], but needing standardization)
    #elif score_type == "word":
    #    return (x - mean_global) / (std_global + 1e-8)

    # Shift-and-Scale Normalization for Sentence Score (Handling condensed range)
    #elif score_type == "sentence":
    elif score_type in ["sentence", "word"]:
        c = 0.02  # Small constant to spread values within [0,1]
        return np.clip((x - mean_global + c) / (2 * c), 0, 1)
    
    # No normalization if undefined
    else:
        return x

# Compute normalization per score type
def normalize_score_old(x, score_type, method="min-max", global_stats=None, global_scores=None):
    """
    Normalize score values based on the selected method using global statistics.
    
    Parameters:
    - x: score value(s) to be normalized.
    - score_type: type of score (determines normalization method).
    - method: default normalization method if not predefined.
    - global_stats: dictionary with precomputed statistics (min, max, mean, std).
    - global_scores: alternative way to compute global statistics dynamically.
    
    Returns:
    - Normalized score value(s).
    """
    
    # Define normalization method based on score type
    if score_type in ["freq", "word", "cos"]:
        method = "min-max"
    elif score_type in ["bert", "sentence", "lm"]:
        method = "z-score"
    elif score_type in ["len"]:
        method = "log"
    else:
        method = None

    """
    if score_type in ["freq", "word", "cos", "bert", "lm"]:
        method = "min-max"
    elif score_type in ["sentence"]:
        method = "z-score"
    elif score_type in ["len"]:
        method = "log"
    else:
        method = None
    """
        
    # Ensure `global_score_array` is a NumPy array
    global_score_array = np.array(global_scores) if global_scores is not None else np.array(x)

    # Compute statistics if not provided
    if global_stats is None:
        min_global = np.min(global_score_array)
        max_global = np.max(global_score_array)
        mean_global = np.mean(global_score_array)
        std_global = np.std(global_score_array)
    else:
        min_global = global_stats.get("min", np.min(global_score_array))
        max_global = global_stats.get("max", np.max(global_score_array))
        mean_global = global_stats.get("mean", np.mean(global_score_array))
        std_global = global_stats.get("std", np.std(global_score_array))

    # Prevent division by zero for std
    std_global = max(std_global, 1e-8)

    # Apply normalization
    if method == "min-max":
        return (x - min_global) / (max_global - min_global + 1e-8)  # Avoid zero division
    elif method == "z-score":
        return (x - mean_global) / std_global
    elif method == "log":
        return np.log1p(x)  # log(1 + x) for stability
    else:
        return x  # No normalization

def get_content_word_indices(sentence):
    """Returns indices of content words (Nouns, Verbs, Adjectives)."""
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    content_word_indices = [
        i for i, (_, tag) in enumerate(pos_tags)
        if tag.startswith(("NN", "VB", "JJ"))  # Nouns, Verbs, Adjectives
    ]
    return content_word_indices

def clean_tokens(token):
    token_wo_occ = token.split("_[")[0] if "_[" in token else token
    token_clean = token_wo_occ[2:] if token_wo_occ.startswith("##") else token_wo_occ
    return token_clean



"""
# Normalization functions to scale ranking scores
# Min-Max Scaling (for scores that are not heavily skewed)
def min_max_scale(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# Z-score Normalization (optional)
def z_score_norm(x):
    return (x - np.mean(x)) / np.std(x)

# IQR Normalization (for length-based scores)
def iqr_norm(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x - np.median(x)) / iqr

# Log Scaling (for LM Score)
def log_scale(x):
    return np.log1p(x)  # log(1 + x) prevents log(0)
"""
