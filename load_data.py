import pandas as pd
import re
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional

def load_data_stats(word_stats_path):
    df_subtlex = pd.read_excel(f"{word_stats_path}SUBTLEX_frequency.xlsx")
    subtlex_words = []
    for i in range(df_subtlex.shape[0]):
        subtlex_words.append(df_subtlex.loc[i,'Word'])
    df_subtlex.set_index('Word',inplace=True)
    #print("Subtlex Words: \n", df_subtlex)

    df_law = pd.read_excel(f"{word_stats_path}/zpf_2.xlsx")
    law_words = []
    for i in range(df_law.shape[0]):
        law_words.append(df_law.loc[i,'Word'])
    df_law.set_index('Word',inplace=True)
    #print("Law Words: \n", df_law)

    eng_file = open(f"{word_stats_path}/english_words.txt", 'r')
    Lines = eng_file.readlines()

    count = 0
    eng_words = []
    # Strips the newline character
    for line in Lines:
        line = re.sub('\n', '', line)
        count += 1
        eng_words.append(line)
    longer_eng_file = open(f"{word_stats_path}/longer_english_words.txt", 'r')
    Lines2 = longer_eng_file.readlines()   

    longer_eng_words = []
    count2 = 0 
    for line2 in Lines2:
        line2 = re.sub('\n', '', line2)
        count2 += 1
        longer_eng_words.append(line2)

    eng_words += longer_eng_words
    eng_words = list(set(eng_words))

    for i in eng_words:
        if len(i) <= 2:
            eng_words.remove(i)

    f = open(f"{word_stats_path}/edited_complex_words_combined.txt",'r',encoding='ascii',errors="ignore")
    complex_words_string = f.read()
    complex_words_string = re.sub(', ', ',', complex_words_string)
    complex_words_string = complex_words_string.lower()
    complex_words = complex_words_string.split(',')
    complex_words = sorted(complex_words, key=len,reverse= True)
    f.close()

    print("Check 2: complex words obtained")
    return df_subtlex, df_law, eng_words, complex_words

def load_legal_sentences(input_path):
    input_file = open(input_path, "r", encoding="utf-8", errors="ignore")
    legal_sentences = input_file.readlines()
    return legal_sentences

def load_cwi_data(batch_size = 1, file_path: Optional[str] = None, dataset: Optional[Dataset] = None, collate_fn = None):
    """
    Loads masked sentence dataset and returns a PyTorch DataLoader.
    """
    if not(dataset):
        assert (file_path is not None)
        dataset = MaskedSentenceDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader

def custom_collate_fn(batch):
    """
    Custom collate function to correctly batch tensor-based masked_inputs 
    while keeping non-tensor elements as lists.
    
    Args:
        batch: List of tuples (masked_text, masked_inputs, words_found, original_text, cleaned_text)
    
    Returns:
        - List of masked_texts (unchanged)
        - Dictionary of padded masked_inputs tensors
        - List of words_found (unchanged)
        - List of tokens_found (unchanged)
        - List of original_texts (unchanged)
        - List of cleaned_texts (unchanged)
    """
    # Unpack batch elements
    masked_texts = [item["masked_text"] for item in batch]
    masked_inputs_list = [item["masked_inputs"] for item in batch]
    words_found_list = [item["words_found"] for item in batch]
    tokens_found_list = [item["tokens_found"] for item in batch]
    original_texts = [item["original_text"] for item in batch]
    cleaned_texts = [item["cleaned_text"] for item in batch]
    
    ## Convert masked_inputs to a dictionary with batched tensors
    batched_inputs = {key: torch.stack([m[key].squeeze(0) for m in masked_inputs_list], dim=0)
                      for key in masked_inputs_list[0].keys()}

    ## Keep non-tensor elements as lists
    #return list(masked_texts), batched_inputs, list(words_found_list), list(original_texts), list(cleaned_texts)

    batch_dict = {
        "masked_texts_batch": masked_texts,
        "masked_inputs_batch": batched_inputs,
        "words_found_batch": words_found_list,
        "tokens_found_batch": tokens_found_list,
        "original_texts_batch": original_texts,
        "cleaned_texts_batch": cleaned_texts,
    }
    return batch_dict

class MaskedSentenceDataset(Dataset):
    def __init__(self, input_path = None, masked_data = None):
        if masked_data is None:
            print("no masked_data supplied, loading from original file...")
            assert (input_path is not None)
        self.masked_texts, self.masked_inputs_list, self.words_found_list, self.tokens_found_list, self.original_texts, self.cleaned_texts, self.dataset_length = \
            self.load_masked_data(masked_file_path=input_path, masked_data=masked_data)

    def load_masked_data(self, masked_file_path=None, masked_data=None):
        """
        Loads stored CWI masked data and processes it in batches.
        """
        if masked_data:
            masked_data = masked_data
        else:
            with open(masked_file_path, "rb") as f:
                masked_data = pickle.load(f)
        
        #print("masked_data")
        #print(masked_data)
        #masked_data = masked_data[:5]
        masked_texts, masked_inputs_list, words_found_list, tokens_found_list, original_texts, cleaned_texts = [], [], [], [], [], []
        # Extract masked text
        for data in masked_data:
            masked_texts.append(data["masked_text"])
            masked_inputs_list.append(data["masked_inputs"])
            words_found_list.append(data["words_found"])
            tokens_found_list.append(data["tokens_found"])
            original_texts.append(data["original_text"])
            cleaned_texts.append(data["cleaned_text"])
        return masked_texts, masked_inputs_list, words_found_list, tokens_found_list, original_texts, cleaned_texts, len(masked_data)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        return {
            "masked_text": self.masked_texts[idx],
            "masked_inputs": self.masked_inputs_list[idx],
            "words_found": self.words_found_list[idx],
            "tokens_found": self.tokens_found_list[idx],
            "original_text": self.original_texts[idx],
            "cleaned_text": self.cleaned_texts[idx],
        }
    
    def getalldatainstances(self):
        return {
            "masked_texts": self.masked_texts,
            "masked_inputs_list": self.masked_inputs_list,
            "words_found_list": self.words_found_list,
            "tokens_found_list": self.tokens_found_list,
            "original_texts": self.original_texts,
            "cleaned_texts": self.cleaned_texts,
        }
