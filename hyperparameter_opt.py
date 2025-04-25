from numpy import dot
from numpy.linalg import norm
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer,BertForMaskedLM
import pandas as pd
import torchtext
import re
import os
import scipy
import copy
#import language_tool_python
from nltk.stem import WordNetLemmatizer 
from readability import Readability
import nltk
import spacy

from load_data import load_data_stats, load_cwi_data, custom_collate_fn, MaskedSentenceDataset
from construct_masked_lm import process_and_store_cwi
from suggestion_generator import run_batched_suggestion_generation, suggestion_filtering
from substitution_ranker import score_substitutions, run_substitution_ranking
from bayesian_opt import optimize_hyperparameters
from utils import ner_checker, load_sentence_model

import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="caselaw", choices=["caselaw", "supreme", "all"])
parser.add_argument("--use_training_data", type=bool, default=False)
parser.add_argument("--gpu_device", default="cuda:0", choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"])
args = parser.parse_args()
dataset = args.dataset
gpu_device = args.gpu_device
use_training_data = args.use_training_data
print("gpu_device: ", gpu_device)
print("dataset: ", dataset)
if dataset == "caselaw":
    use_training_data = False #already using full data

#tool = language_tool_python.LanguageTool('en-US')
lemmatizer = WordNetLemmatizer()

import time
start = time.process_time()

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')  # Check if already downloaded
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')  # Check if already downloaded
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('tokenizers/punkt/punkt.tab')  # Correct path for punkt.tab
except LookupError:
    nltk.download('punkt')  # Downloading 'punkt' usually includes 'punkt.tab'

#nltk.download('punkt_tab')
glove = torchtext.vocab.GloVe(name="6B", dim=300) # trained on Wikipedia 2014 corpus
#fasttext = torchtext.vocab.FastText()

nlp_ner = spacy.load("en_core_web_sm") #nlp model
#nlp_ner = spacy.load("en_core_web_lg") #

device = torch.device(gpu_device if torch.cuda.is_available() else "cpu")

tokenizer = legal_base_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = legal_base_model = BertForMaskedLM.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_base_model.to(device)
num_suggestions = 100
batch_size = 8
#num_suggestions = 10

print("Check 1: Imports successful \n")

word_stats_path = "./word_stats/"
df_subtlex, df_law, eng_words, complex_words = load_data_stats(word_stats_path)

print("Check 2: Data loading complete.\n")

if dataset == "caselaw":
    complex_path = "./dataset/caselaw/caselaw_500.txt"
    complex_paths = [complex_path]
    masked_output_path = "./dataset/caselaw/caselaw_500_masked.pkl"
    #simple_path = "./hyperparam_opt/caselaw/uslt_noss_caselaw_test.txt"
elif dataset == "supreme":
    if use_training_data:
        complex_path = "./dataset/supreme_court/train/supreme_train_org.txt"
        complex_paths = [complex_path]
        masked_output_path = "./dataset/supreme_court/val/supreme_masked_valtrain_org.txt"
        simple_path = "./hyperparam_opt/supreme_court/val/uslt_noss_supreme_valtrain.txt"
    else:
        complex_path = "./dataset/supreme_court/val/supreme_val_org.txt"
        complex_paths = [complex_path]
        masked_output_path = "./dataset/supreme_court/val/supreme_masked_val_org.txt"
        simple_path = "./hyperparam_opt/supreme_court/val/uslt_noss_supreme_val.txt"
elif dataset == "all":
    if use_training_data:
        complex_path1 = "./dataset/supreme_court/train/supreme_train_org.txt"
        complex_path2 = "./dataset/caselaw/caselaw_500.txt"
        complex_paths = [complex_path1, complex_path2]
        masked_output_path = "./dataset/all/all_masked_train_org.txt"
        simple_path = "./hyperparam_opt/all/all_noss_supreme_train.txt"
    else:
        complex_path1 = "./dataset/supreme_court/val/supreme_val_org.txt"
        complex_path2 = "./dataset/caselaw/caselaw_500.txt"
        complex_paths = [complex_path1, complex_path2]
        masked_output_path = "./dataset/all/all_masked_val_org.txt"
        simple_path = "./hyperparam_opt/all/all_noss_supreme_val.txt"

if os.path.isfile(masked_output_path) and 1==0:
#if os.path.isfile(masked_output_path):
    print("Loading the masked dataset...")
    masked_dataset = MaskedSentenceDataset(input_path=masked_output_path)
else:
    print("Creating the masked dataset...")
    complex_lines = []
    for complex_path in complex_paths:
        f = open(complex_path, 'r', encoding = "utf-8", errors="ignore")
        #for line in f.readlines()[:20]:
        for line in f:
            complex_lines.append(line.strip("\n"))
        f.close()
    masked_data = process_and_store_cwi(complex_lines, tokenizer, complex_words, ner_checker, nlp_ner, masked_output_path, return_data=True)
    masked_dataset = MaskedSentenceDataset(masked_data=masked_data)

full_masked_data = masked_dataset.getalldatainstances()
all_cleaned_texts, all_original_texts, all_words_found = full_masked_data["cleaned_texts"], full_masked_data["original_texts"], full_masked_data["words_found_list"]
dataloader = load_cwi_data(batch_size, dataset=masked_dataset, collate_fn=custom_collate_fn)
all_suggestions = run_batched_suggestion_generation(dataloader, model, tokenizer, num_suggestions, device)
print("all_suggestions")
print(all_suggestions)

sentence_model = load_sentence_model(model, device)

if num_suggestions <= 5:
    num_suggestions_filtered = num_suggestions  # Keep all candidates if we already have very few
elif num_suggestions <= 10:
    num_suggestions_filtered = max(3, int(0.5 * num_suggestions))  # Keep 60% but at least 3
elif num_suggestions <= 20:
    num_suggestions_filtered = max(5, int(0.35 * num_suggestions))  # Keep 50% but at least 5
else:
    num_suggestions_filtered = max(8, int(0.2 * num_suggestions))  # Keep 30% but at least 10
    #num_suggestions_filtered = 30

#num_suggestions_filtered = num_suggestions
print("num_suggestions:", num_suggestions)
print("num_suggestions_filtered:", num_suggestions_filtered)
#num_suggestions_filtered = 20 # for no cos filtering, assign num_suggestions_filtered = num_suggestions
all_filtered_suggestions = suggestion_filtering(all_suggestions, all_cleaned_texts, 
                                                all_words_found, 
                                                df_subtlex, complex_words, eng_words, 
                                                glove, 
                                                tokenizer, 
                                                num_suggestions_filtered)

all_suggestion_scores = score_substitutions(all_filtered_suggestions, 
                                            df_subtlex, complex_words, eng_words, 
                                            all_cleaned_texts, 
                                            model, sentence_model, tokenizer, glove,
                                            device)

# Call the function
optimal_raw_weights = optimize_hyperparameters(
    all_suggestion_scores, df_subtlex, complex_words, eng_words, 
    all_cleaned_texts, all_original_texts, model, tokenizer, 
    glove, device, n_trials=1000
)
optimal_weights = {}
optimal_raw_weights_sum = 0
for key in optimal_raw_weights.keys():
    optimal_raw_weights_sum += optimal_raw_weights[key]
for key in optimal_raw_weights.keys():
    optimal_weights[key] = optimal_raw_weights[key]/optimal_raw_weights_sum
print("Optimal Weights:", optimal_weights)
#write_simplifications_to_file(all_simple_sentences, simple_path, complex_path)

end = time.process_time()
print("Elapsed time using process_time()", (end - start) * 10**3, "ms.")
