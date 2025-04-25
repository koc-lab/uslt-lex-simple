import torch
import re
import copy
import pickle

def mask_complex_words_word(text, tokenizer, complex_word_list, ner_checker, nlp_ner):
    """
    Replaces complex words in the text with [MASK] tokens for prediction.
    Ensures that multi-word complex words are correctly handled and checks NER.
    Applies masking at the token level rather than masking the words.
    """
    words_found = []  # To track words being replaced
    tokens_found = [] # To track tokens being replaced
    replaced_tokens = []  # To track token replacements
    replaced_token_ids = [] # To track the ids of tokens being replaced
    selection = []  # To track the indices of words that are replaced
    
    # Tokenize the original text and clean it
    cleaned_text = re.sub('[^A-Za-z]+', ' ', text)  # Remove non-alphabetical characters
    text_list = cleaned_text.split()
    masked_text_list = copy.deepcopy(text_list)

    cleaned_text_tokens = tokenizer.tokenize(cleaned_text)
    text_tokens = tokenizer.tokenize(text)
    masked_text_tokens = copy.deepcopy(cleaned_text_tokens)
    masked_text = copy.deepcopy(cleaned_text)
    #masked_text_tokens = copy.deepcopy(text_tokens)
    #masked_text = copy.deepcopy(text)
    
    print("cleaned_text")
    print(cleaned_text)
    print("text_list")
    print(text_list)

    complex_words_sentence = []
    max_complex_word_len = 0

    # Iterate over each word in word_list to find complex words in text
    #for word in word_list:
    #    if word in cleaned_text:
    #prefix_suffix_list = ["##ment", "##ly", "##ir", "##ness", "##ity", "##ship", "##ful", "##less"]
    for complex_word in complex_word_list:
        if complex_word in text.lower() or complex_word in cleaned_text.lower():
            cleaned_complex_word = re.sub('[^A-Za-z]+', ' ', complex_word)
            cleaned_complex_word_split = cleaned_complex_word.split()
            if (len(complex_word.split()) == 1 and not(ner_checker(nlp_ner, text, complex_word))) or (len(complex_word.split()) > 1):
                complex_words_sentence.append(complex_word)
                if len(cleaned_complex_word_split) > max_complex_word_len:
                    max_complex_word_len = len(cleaned_complex_word_split)
    
    print("complex_words_sentence")
    print(complex_words_sentence)
    
    for size in reversed(range(1, max_complex_word_len+1)): #mask multi-word expressions first in case individual words are also in complex words
        i = 0
        while i <= len(masked_text_list) - size:
            target_word_split = masked_text_list[i:i + size]
            target_word = " ".join(target_word_split)
            # Check if the tokens at the current position match the word to mask
            if target_word.lower() in complex_words_sentence:
                words_found.append(target_word)
                for complex_token in target_word_split:
                    tokens_found.append(complex_token)
                # Replace each token in the word with [MASK]
                masked_text_list[i:i + size] = ["[MASK]"] * size
                # Move the index forward to continue searching
                i += size
            else:
                i += 1

    masked_text = " ".join(masked_text_list)

    return masked_text, cleaned_text, words_found, tokens_found, selection


def mask_complex_words_word_old(text, tokenizer, complex_word_list, ner_checker, nlp_ner):
    """
    Replaces complex words in the text with [MASK] tokens for prediction.
    Ensures that multi-word complex words are correctly handled and checks NER.
    Applies masking at the token level rather than masking the words.
    """
    words_found = []  # To track words being replaced
    tokens_found = [] # To track tokens being replaced
    replaced_tokens = []  # To track token replacements
    replaced_token_ids = [] # To track the ids of tokens being replaced
    selection = []  # To track the indices of words that are replaced
    
    # Tokenize the original text and clean it
    cleaned_text = re.sub('[^A-Za-z]+', ' ', text)  # Remove non-alphabetical characters
    text_list = cleaned_text.split()
    masked_text_list = copy.deepcopy(text_list)

    cleaned_text_tokens = tokenizer.tokenize(cleaned_text)
    text_tokens = tokenizer.tokenize(text)
    masked_text_tokens = copy.deepcopy(cleaned_text_tokens)
    masked_text = copy.deepcopy(cleaned_text)
    #masked_text_tokens = copy.deepcopy(text_tokens)
    #masked_text = copy.deepcopy(text)
    
    print("cleaned_text")
    print(cleaned_text)
    print("text_list")
    print(text_list)

    # Iterate over each word in word_list to find complex words in text
    #for word in word_list:
    #    if word in cleaned_text:
    prefix_suffix_list = ["##ment", "##ly", "##ir"]
    #prefix_suffix_list = ["##ment", "##ly", "##ir", "##ness", "##ity", "##ship", "##ful", "##less"]
    for l in range(len(text_list)):
        word = text_list[l]
        word_tokenized = tokenizer.tokenize(word)
        if word.lower() in complex_word_list or any(token.lstrip("#") in complex_word_list and token not in prefix_suffix_list for token in word_tokenized):
            # Check if the word is a multi-word phrase or a complex word based on NER
            #if (len(word.split()) >= 2 and word.split()[0] in complex_word_list) or (word in text_list and ner_checker(nlp_ner, text, word)):
            if not(ner_checker(nlp_ner, text, word)): # makes sure that the word is not recognized as an entity
                word_tokens = word_tokenized
            else:
                continue  # Skip non-complex words or NER entities
                
            # Replace the complex word in (cleaned_)text with [MASK]
            #text = re.sub(r'\b' + re.escape(word) + r'\b', '[MASK]', text)
            masked_text_list[l] = "[MASK]"

            # Step 5: Convert masked tokens back to a string
            #print("masked_text_tokens")
            #print(masked_text_tokens)
            masked_text = " ".join(masked_text_list)
            #print("masked_text")
            #print(masked_text)
            words_found.append(word)
            #tokens_found.extend(word_tokens)
            tokens_found.append(word)
            #print("words_found")
            #print(words_found)
                
    print("words_found")
    print(words_found)
    print("tokens_found")
    print(tokens_found)

    return masked_text, cleaned_text, words_found, tokens_found, selection

def mask_complex_words_token(text, tokenizer, complex_word_list, ner_checker, nlp_ner):
    """
    Replaces complex words in the text with [MASK] tokens for prediction.
    Ensures that multi-word complex words are correctly handled and checks NER.
    Applies masking at the token level rather than masking the words.
    """
    words_found = []  # To track words being replaced
    tokens_found = [] # To track tokens being replaced
    replaced_tokens = []  # To track token replacements
    replaced_token_ids = [] # To track the ids of tokens being replaced
    selection = []  # To track the indices of words that are replaced
    
    # Tokenize the original text and clean it
    cleaned_text = re.sub('[^A-Za-z]+', ' ', text)  # Remove non-alphabetical characters
    text_list = cleaned_text.split()

    cleaned_text_tokens = tokenizer.tokenize(cleaned_text)
    text_tokens = tokenizer.tokenize(text)
    masked_text_tokens = copy.deepcopy(cleaned_text_tokens)
    masked_text = copy.deepcopy(cleaned_text)
    #masked_text_tokens = copy.deepcopy(text_tokens)
    #masked_text = copy.deepcopy(text)
    
    print("cleaned_text")
    print(cleaned_text)
    print("text_list")
    print(text_list)

    # Iterate over each word in word_list to find complex words in text
    #for word in word_list:
    #    if word in cleaned_text:
    prefix_suffix_list = ["##ment", "##ly", "##ir"]
    #prefix_suffix_list = ["##ment", "##ly", "##ir", "##ness", "##ity", "##ship", "##ful", "##less"]
    for word in text_list:
        word_tokenized = tokenizer.tokenize(word)
        if word.lower() in complex_word_list or any(
            token.lstrip("#") in complex_word_list and token not in prefix_suffix_list for token in word_tokenized):
            # Check if the word is a multi-word phrase or a complex word based on NER
            #if (len(word.split()) >= 2 and word.split()[0] in complex_word_list) or (word in text_list and ner_checker(nlp_ner, text, word)):
            if not(ner_checker(nlp_ner, text, word)): # makes sure that the word is not recognized as an entity
                word_tokens = word_tokenized
            else:
                continue  # Skip non-complex words or NER entities
                
            # Replace the complex word in (cleaned_)text with [MASK]
            #text = re.sub(r'\b' + re.escape(word) + r'\b', '[MASK]', text)
            i = 0
            while i <= len(masked_text_tokens) - len(word_tokens):
                # Check if the tokens at the current position match the word to mask
                if masked_text_tokens[i:i + len(word_tokens)] == word_tokens:
                    # Replace each token in the word with [MASK]
                    masked_text_tokens[i:i + len(word_tokens)] = ["[MASK]"] * len(word_tokens)
                    # Move the index forward to continue searching
                    i += len(word_tokens)
                else:
                    i += 1

            # Step 5: Convert masked tokens back to a string
            #print("masked_text_tokens")
            #print(masked_text_tokens)
            masked_text = tokenizer.convert_tokens_to_string(masked_text_tokens)
            #print("masked_text")
            #print(masked_text)
            words_found.append(word)
            tokens_found.extend(word_tokens)
            #print("words_found")
            #print(words_found)
                
    print("words_found")
    print(words_found)
    print("tokens_found")
    print(tokens_found)

    return masked_text, cleaned_text, words_found, tokens_found, selection

def create_masked_lm(tokenizer, original_text, masked_text):
    """
    Creates masked language model inputs for BERT-based models.
    
    Args:
        tokenizer: Tokenizer instance.
        original_text (str): The unaltered input text.
        selection (list[int]): List of token indices to be masked.
        masked_text (str): Text where words are replaced with [MASK].
        device (str): The device to run on (default: 'cuda').
        
    Returns:
        inputs (dict): Tokenized input for BERT.
        masked_word_ids (list): List of token IDs including masked ones.
    """
    #print("original_text")
    #print(original_text)

    combined_text = f"{original_text} {tokenizer.sep_token} {masked_text}"
    
    # Tokenize masked text
    masked_inputs = tokenizer(
        combined_text, 
        return_tensors='pt', 
        truncation=True,
        max_length=512,
        padding="max_length")
    #print("masked_inputs")
    #print(masked_inputs)
    
    # Tokenized words (for debugging)
    masked_tokens = tokenizer.tokenize(masked_text)
    #print("masked_tokens")
    #print(masked_tokens)
    
    # Clone input IDs for labels (for loss calculation)
    masked_inputs['labels'] = masked_inputs.input_ids.clone().detach()
        
    # Tokenize original text (handling truncation)
    original_inputs = tokenizer(
        original_text, 
        return_tensors='pt', 
        truncation=True, 
        max_length=512,
        padding="max_length"
        #max_length=512 - len(masked_tokens)  # Ensuring BERT limit
    )
    original_inputs['labels'] = original_inputs.input_ids.clone().detach()
    
    #print("original_inputs")
    #print(original_inputs)

    # Concatenating the two inputs along the sequence dimension
    #for key in masked_inputs:
    #    masked_inputs[key] = torch.cat((original_inputs[key], masked_inputs[key][:,1:]), dim=1)

    #print("masked_inputs")
    #print(masked_inputs)
        
    # Extract masked word IDs
    masked_word_ids = masked_inputs['labels'][0].tolist()
    #print("masked_word_ids")
    #print(masked_word_ids)
    
    return masked_inputs, masked_word_ids

def process_cwi(original_text, tokenizer, word_list, ner_checker, nlp_ner, diff_repeating_occ=True):
    """
    Main function to process Complex Word Identification (CWI).
    This will identify complex words and replace them with [MASK] for further prediction.
    """
    
    # Process the text by replacing complex words with [MASK]
    #masked_text, cleaned_text, words_found, tokens_found, selection = mask_complex_words_token(
    #    original_text, tokenizer, word_list, ner_checker, nlp_ner
    #)
    masked_text, cleaned_text, words_found, tokens_found, selection = mask_complex_words_word(
        original_text, tokenizer, word_list, ner_checker, nlp_ner
    )
    print("masked_text")
    print(masked_text)
    
    if diff_repeating_occ: #labels repeating complex words differently
        #words_found = differentiate_repeating_complex_words(words_found)
        tokens_found = differentiate_repeating_complex_words(tokens_found)
    
    # Prepare masked input for BERT processing
    cleaned_inputs = tokenizer(cleaned_text, return_tensors='pt')
    #masked_inputs = tokenizer(masked_text, return_tensors='pt')
    
    # Further processing can be added here (e.g., model inference)

    #masked_inputs, masked_word_ids = create_masked_lm(tokenizer, original_text, masked_text)
    masked_inputs, masked_word_ids = create_masked_lm(tokenizer, cleaned_text, masked_text)

    #print("masked_text")
    #print(masked_text)
    # Return necessary information
    return {
        'original_text': original_text,
        'cleaned_text': cleaned_text,
        'cleaned_inputs': cleaned_inputs,
        'masked_text': masked_text,
        'masked_inputs': masked_inputs,
        'masked_word_ids': masked_word_ids,
        'words_found': words_found,
        'tokens_found': tokens_found,
        'selection': selection
    }

def differentiate_repeating_complex_words(replaced_words):
    """
    labels repeating complex words as "{word}_[1]", "{word}_[2]" etc
    to handle multiple occurences of the complex word separately
    """
    for word in replaced_words:
        if replaced_words.count(word) > 1:
            for word_count in range(replaced_words.count(word)):
                idx = replaced_words.index(word)
                replaced_words[idx] = word + f"_[{word_count+1}]"
    return replaced_words

def process_and_store_cwi(dataset, tokenizer, complex_word_list, ner_checker, nlp_ner, output_path, return_data=False):
    """
    Runs CWI on the dataset and stores the masked outputs for later use.
    """
    masked_data = []
    
    i = 0
    for sentence in dataset:
        i += 1
        print(f"Sentence {i}/{len(dataset)}...")
        cwi_output = process_cwi(sentence, tokenizer, complex_word_list, ner_checker, nlp_ner)
        masked_data.append(cwi_output)

    # Save to file (Pickle for structured data)
    with open(output_path, "wb") as f:
        pickle.dump(masked_data, f)
    
    print(f"Masked CWI data saved to {output_path}")
    #print(f"Masked CWI data NOT! saved to {output_path}")
    return masked_data