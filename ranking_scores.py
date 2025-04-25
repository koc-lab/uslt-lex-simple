import torch
import copy
import re
import nltk
import torch.nn.functional as F
from transformers import AutoModel, BertForMaskedLM
from utils import get_content_word_indices, place_suggestion_into_sentence_word, place_suggestion_into_sentence_token

def compute_frequency_score(word, df_subtlex):
    """ Calculate frequency score based on Zipf-value or default if missing. """
    try:
        element = df_subtlex.loc[word, 'Zipf-value']
        #return float(element) * weight_freq
        return float(element)
    except KeyError:
        # Default score if word is not found
        #return float(1.5) * weight_freq
        return float(1.5)

def compute_length_score(word):
    """ Calculate length score based on word length. """
    #return weight_len * (1 / len(word)) ** 3.78
    return (1 / len(word)) ** 3.78

def compute_cosine_similarity(word1, word2, glove, tokenizer=None):
    """ Calculate cosine similarity between two words using GloVe vectors. """
    try:
        cos_sim = float(torch.cosine_similarity(glove[word1].unsqueeze(0), glove[word2].unsqueeze(0)))
        return cos_sim
    except KeyError:
        print("KeyError occured while computing cosine_similarity...")
        raise KeyError
        return 0.0

def compute_cosine_similarity_token(word1, word2, glove, tokenizer=None):
    """ Calculate cosine similarity between two words using GloVe vectors. """
    try:
        tokens_cos_sim = []
        if tokenizer is not None:
            target_tokens = tokenizer.tokenize(word1)
            if len(target_tokens) != 1:
                target_tokens.append(word1)
        else:
            target_tokens = [word1]
        for target_token in target_tokens:
            cos_sim_token = float(torch.cosine_similarity(glove[target_token].unsqueeze(0), glove[word2].unsqueeze(0)))
            tokens_cos_sim.append(cos_sim_token)
        cos_sim = max(tokens_cos_sim)
        #return cos_sim * weight_cos
        return cos_sim
    except KeyError:
        print("KeyError occured while computing cosine_similarity...")
        raise KeyError
        return 0.0

def calculatelmloss(original_text, target_word, suggestion, model, tokenizer, target_occurrence_count=1, mask_window=2, max_length=512, device="cuda"):
    """
    Calculates the language model loss by masking words around the target word's substitution
    and evaluating the model's ability to predict them.
    
    Parameters:
    - original_text (str): The original sentence (cleaned version).
    - target_word (str): The complex word (token if there are subword tokens) in the sentence.
    - suggestion (str): The suggested substitution for the target word.
    - model: The language model to compute the loss.
    - tokenizer: The tokenizer for the language model.
    - mask_window (int): The number of words to mask before and after the target word (default is 2).
    - max_length (int): Maximum sequence length for the tokenizer (default is 256).

    Returns:
    - total_loss (float): The average loss for the masked words.
    """

    #modified_text, suggestion_index = place_suggestion_into_sentence_token(original_text, target_word, suggestion, tokenizer, target_occurrence_count)
    modified_text, suggestion_index = place_suggestion_into_sentence_word(original_text, target_word, suggestion, target_occurrence_count)
    # Tokenize the text
    inputs = tokenizer.encode_plus(
        modified_text, 
        return_tensors="pt", 
        add_special_tokens=True, 
        truncation=True, 
        padding='max_length', 
        return_attention_mask=True, 
        max_length=max_length
    )
    inputs = inputs.to(device)
    labels = copy.deepcopy(inputs['input_ids'])

    # Split the text into individual words
    clean_text = re.sub('[^A-Za-z]+', ' ', modified_text)
    text_list = clean_text.split()

    total_loss = 0.0

    masked_word_count = 0
    masked_word_list = text_list[:]
    # Mask words within the window and compute loss
    for i in range(-mask_window, mask_window + 1):
        if i == 0:
            continue  # Skip masking the suggestion itself

        # Create a copy of the word list and mask the word at the specified position
        #masked_word_list = text_list[:]
        #masked_word_list = text_tokens[:]
        try:
            masked_word_list[suggestion_index + i] = '[MASK]'
            masked_word_count += 1
        except IndexError:
            continue  # Skip if the index is out of range

    # Join the masked word list back into a string
    masked_text = ' '.join(masked_word_list)
    #masked_text = tokenizer.convert_tokens_to_string(masked_word_list)

    # Tokenize the masked text
    masked_inputs = tokenizer.encode_plus(
        masked_text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        max_length=max_length
    )
    masked_inputs = masked_inputs.to(device)

    # Update the inputs with masked input IDs
    inputs['input_ids'] = masked_inputs['input_ids']

    # Set non-masked tokens to -100 in the labels (ignored by the loss function)
    labels = inputs['input_ids'].clone()
    labels[inputs['input_ids'] != tokenizer.mask_token_id] = -100

    # Compute the model's output and loss
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        token_type_ids=inputs.get('token_type_ids', None),
        labels=labels
    )
    loss = outputs.loss
    total_loss += float(loss)

    # Reset the labels for the next iteration
    labels = copy.deepcopy(inputs['input_ids'])

    # Return the average loss over the masked words
    return total_loss / (masked_word_count)

def compute_lm_perplexity_loss(original_text, target_word, suggestion, model, tokenizer, target_occurrence_count, device):
    """ Calculate language model loss for a word substitution. """
    #if weight_lm == 0:
    #    return 0.0

    try:
        total_loss = calculatelmloss(original_text, target_word, suggestion, model, tokenizer, target_occurrence_count=target_occurrence_count, device=device)
        if total_loss != 0:
            #return (1 + 0.5 / total_loss + 0.5) * weight_lm
            #return (1 + 0.5 / total_loss + 0.5)
            return 1/total_loss
    #except Exception:
    except:
        print("Something went wrong...")
        print(f"LM loss could not be computed for target word: **{target_word}** and suggestion: **{suggestion}**")
        raise IndexError
    return 0.0

def compute_contextual_fit_score(original_text, target_word, suggestion, model, tokenizer, target_occurrence_count=1, context_window=4, max_length=256, device="cuda"):
    """
    Computes the Contextual Fit Score by measuring the probability of the next 'context_window' words
    after the suggested simplification.

    Parameters:
    - original_text (str): The original sentence.
    - target_word (str): The complex word.
    - suggestion (str): The suggested simplification.
    - model: Pretrained masked language model (e.g., BERT, RoBERTa).
    - tokenizer: Tokenizer for the model.
    - target_occurrence_count (int): The occurrence of the target word to replace.
    - context_window (int): The number of words to consider after the simplification.
    - max_length (int): Maximum sequence length for the tokenizer.
    - device (str): Device to run the model on (default: 'cuda').

    Returns:
    - context_fit_score (float): Mean probability of next words appearing in context.
    """

    #modified_text, suggestion_index = place_suggestion_into_sentence_token(original_text, target_word, suggestion, tokenizer, target_occurrence_count)
    modified_text, suggestion_index = place_suggestion_into_sentence_word(original_text, target_word, suggestion, target_occurrence_count)

    # Tokenize modified sentence
    inputs = tokenizer(modified_text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
    inputs = inputs.to(device)

    # Get tokenized sentence representation
    #tokenized_sentence = tokenizer.tokenize(modified_text)
    tokenized_sentence = original_text.split()

    # Find index of the suggested word
    observed_target_token = 0
    suggestion_index = None
    for idx, token in enumerate(tokenized_sentence):
        if suggestion in token:
            observed_target_token += 1
            if observed_target_token == target_occurrence_count:
                suggestion_index = idx
                break

    if suggestion_index is None:
        print(f"Could not find '{suggestion}' in tokenized text.")
        return 0.0

    # Compute model logits (unnormalized scores)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)  # Shape: [seq_length, vocab_size]

    # Compute probabilities using softmax
    probs = F.softmax(logits, dim=-1)

    # Sum log probabilities for next 'context_window' words
    total_log_prob = 0.0
    count = 0

    for i in range(1, context_window + 1):  # Checking the next 'context_window' words
        next_word_index = suggestion_index + i
        if next_word_index >= len(tokenized_sentence):
            break

        next_token = tokenized_sentence[next_word_index]
        next_token_id = tokenizer.convert_tokens_to_ids(next_token)

        if next_token_id == tokenizer.unk_token_id:
            continue  # Skip unknown tokens

        next_word_prob = probs[next_word_index, next_token_id]
        total_log_prob += torch.log(next_word_prob + 1e-9)  # Avoid log(0)
        count += 1

    if count == 0:
        return 0.0

    # Compute mean log probability
    context_fit_score = total_log_prob.item() / count
    return context_fit_score

from nltk.stem import PorterStemmer
ps = PorterStemmer()

def morphological_simplicity(original, candidate):
    return 1 if ps.stem(original) == ps.stem(candidate) else 0  # Binary match

from Levenshtein import distance as levenshtein_distance

def edit_distance_score(original, candidate):
    return 1 / (1 + levenshtein_distance(original, candidate))

#from transformers import AutoModel, AutoTokenizer

## Load your LegalBERT model
#bert_model_name = "nlpaueb/legal-bert-base-uncased"
#tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
#model = AutoModel.from_pretrained(bert_model_name).eval()

def get_bert_embedding(word, model, tokenizer, device):
    """
    Given a word, returns its BERT contextual embedding.
    
    Args:
    - word (str): The input word.
    - model (BERT model): Pretrained Legal-BERT model.
    - tokenizer (BERT tokenizer): Tokenizer for Legal-BERT.
    
    Returns:
    - torch.Tensor: The word's contextual embedding.
    """
    # Tokenize the word and prepare input
    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=True).to(device)

    with torch.no_grad():  # No gradients needed
        if isinstance(model, BertForMaskedLM):
            outputs = model.bert(**inputs, output_hidden_states=True)  # Get hidden states
        elif isinstance(model, AutoModel):
            outputs = model(**inputs)
        else:
            raise ValueError("Unsupported model type")
        hidden_states = outputs.hidden_states  # Tuple of hidden layers

    # Use the LAST HIDDEN STATE (BERT Base has 12 layers)
    last_hidden_state = hidden_states[-1]  # Shape: (1, seq_len, hidden_dim=768)

    # Get the embedding of the **word token** (excluding special tokens)
    word_embedding = last_hidden_state[:, 1:-1, :]  # Remove [CLS] and [SEP]

    word_embedding = word_embedding.mean(dim=1)  # Return the mean across subwords (if any)

    return word_embedding.squeeze(dim=0)

def bert_word_similarity_score(word1, word2, model, tokenizer, device):
    """Computes cosine similarity between BERT embeddings of two words."""
    embedding1 = get_bert_embedding(word1, model, tokenizer, device)
    embedding2 = get_bert_embedding(word2, model, tokenizer, device)
    
    cos_sim = F.cosine_similarity(embedding1, embedding2, dim=0)
    return cos_sim.item()

## Example usage
#similarity = bert_word_similarity("federal", "government", model, tokenizer)
#print(f"BERT Word Similarity: {similarity}")

# Load a strong general-purpose sentence embedding model
#st_model = SentenceTransformer("all-mpnet-base-v2")

def bert_sentence_similarity(sentence1, sentence2, sentence_model):
    """Computes cosine similarity between two sentence embeddings."""
    embedding1 = sentence_model.encode(sentence1, convert_to_tensor=True)
    embedding2 = sentence_model.encode(sentence2, convert_to_tensor=True)

    cos_sim = sentence_model.similarity(embedding1, embedding2)
    return cos_sim.squeeze().tolist()

## Example Usage
#original_sentence = "The federal government imposed new regulations."
#simplified_sentence = "The national government made new rules."

#similarity = bert_sentence_similarity(original_sentence, simplified_sentence, st_model)
#print(f"BERT Sentence Similarity: {similarity}")

def compute_contextual_word_similarity(original_text, target_word, suggestion, model, tokenizer, target_occurrence_count=1, max_length=512, device="cuda"):
    """
    
    Parameters:
    - original_text (str): The original sentence (cleaned version).
    - target_word (str): The complex word (token if there are subword tokens) in the sentence.
    - suggestion (str): The suggested substitution for the target word.
    - model: The language model to compute the loss.
    - tokenizer: The tokenizer for the language model.
    - mask_window (int): The number of words to mask before and after the target word (default is 2).
    - max_length (int): Maximum sequence length for the tokenizer (default is 256).

    Returns:
    - 
    """

    #modified_text, suggestion_index = place_suggestion_into_sentence_token(original_text, target_word, suggestion, tokenizer, target_occurrence_count)
    modified_text, suggestion_index = place_suggestion_into_sentence_word(original_text, target_word, suggestion, target_occurrence_count)

    # Tokenize the text
    original_inputs = tokenizer.encode_plus(
        original_text, 
        return_tensors="pt", 
        add_special_tokens=True, 
        truncation=True, 
        padding='max_length', 
        return_attention_mask=True, 
        max_length=max_length
    )

    inputs = tokenizer.encode_plus(
        modified_text, 
        return_tensors="pt", 
        add_special_tokens=True, 
        truncation=True, 
        padding='max_length', 
        return_attention_mask=True, 
        max_length=max_length
    )
    original_inputs = original_inputs.to(device)
    inputs = inputs.to(device)

    with torch.no_grad():
        original_outputs = model(**original_inputs, output_hidden_states=True)
        org_hidden_states = original_outputs.hidden_states
        outputs = model(**inputs, output_hidden_states=True)
        out_hidden_states = outputs.hidden_states
    
    # Extract the last hidden state
    org_last_hidden_state = org_hidden_states[-1]  # Shape: (1, seq_len, 768)
    out_last_hidden_state = out_hidden_states[-1]  # Shape: (1, seq_len, 768)

    org_word_embedding = org_last_hidden_state[:, suggestion_index, :]
    out_word_embedding = out_last_hidden_state[:, suggestion_index, :]

    return float(torch.cosine_similarity(org_word_embedding, out_word_embedding))
