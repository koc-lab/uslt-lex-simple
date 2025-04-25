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
from bayesian_opt import evaluate_w_kfold_cross_val
from construct_masked_lm import process_and_store_cwi
from suggestion_generator import run_batched_suggestion_generation, suggestion_filtering
from substitution_ranker import score_substitutions, run_substitution_ranking
from simplify import simplify_batched_text, write_simplifications_to_file
from utils import ner_checker, load_sentence_model

import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_device", default="cuda:0", choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"])
parser.add_argument("--data_split", default="test", choices=["train", "test", "val"])
args = parser.parse_args()
gpu_device = args.gpu_device
data_split = args.data_split
print("gpu_device: ", gpu_device)
print("data_split: ", data_split)

#tool = language_tool_python.LanguageTool('en-US')
lemmatizer = WordNetLemmatizer()

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
#glove = torchtext.vocab.GloVe(name="42B", dim=300) # trained on Wikipedia 2014 corpus
#fasttext = torchtext.vocab.FastText()
glove = torchtext.vocab.FastText(language="en")

nlp_ner = spacy.load("en_core_web_sm") #nlp model
#nlp_ner = spacy.load("en_core_web_lg") #

device = torch.device(gpu_device if torch.cuda.is_available() else "cpu")

tokenizer = legal_base_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = legal_base_model = BertForMaskedLM.from_pretrained("nlpaueb/legal-bert-base-uncased").eval()
from transformers import AutoModelForCausalLM
#model_causal = AutoModelForCausalLM.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_base_model.to(device)
num_suggestions = 100
batch_size = 8
#num_suggestions = 10

print("Check 1: Imports successful \n")

word_stats_path = "./word_stats/"
df_subtlex, df_law, eng_words, complex_words = load_data_stats(word_stats_path)

print("Check 2: Data loading complete.\n")

#complex_path = "./lex-simple-500-dataset/test/supreme_org_test.txt"
if data_split == "test":
    complex_path = "./dataset/supreme_court/test/supreme_org_test.txt"
    masked_output_path = "./dataset/supreme_court/test/supreme_org_masked_test.pkl"
    simple_path = "./output_data/test/uslt_noss_supreme_test.txt"
else:
    complex_path = f"./dataset/supreme_court/{data_split}/supreme_{data_split}_org.txt"
    masked_output_path = f"./dataset/supreme_court/{data_split}/supreme_masked_{data_split}_org.pkl"
    simple_path = f"./output_data/{data_split}/supreme_{data_split}_uslt_noss.txt"

if os.path.isfile(masked_output_path) and 1==0:
#if os.path.isfile(masked_output_path):
    print("Loading the masked dataset...")
    masked_dataset = MaskedSentenceDataset(input_path=masked_output_path)
else:
    print("Creating the masked dataset...")
    f = open(complex_path, 'r', encoding = "utf-8", errors="ignore")
    complex_lines = []
    #for line in f.readlines()[:20]:
    for line in f:
        complex_lines.append(line.strip("\n"))
    f.close()
    masked_data = process_and_store_cwi(complex_lines, tokenizer, complex_words, ner_checker, nlp_ner, masked_output_path, return_data=True)
    masked_dataset = MaskedSentenceDataset(masked_data=masked_data)

full_masked_data = masked_dataset.getalldatainstances()
print("full_masked_data")
print("masked_texts")
print(full_masked_data["masked_texts"])
print("cleaned_texts")
print(full_masked_data["cleaned_texts"])
all_cleaned_texts, all_original_texts, all_words_found = full_masked_data["cleaned_texts"], full_masked_data["original_texts"], full_masked_data["words_found_list"]
dataloader = load_cwi_data(batch_size, dataset=masked_dataset, collate_fn=custom_collate_fn)
all_suggestions = run_batched_suggestion_generation(dataloader, model, tokenizer, num_suggestions, device)
print("all_suggestions")
print(all_suggestions)
sentence_model = load_sentence_model(model, device)
if num_suggestions <= 5:
    num_suggestions_filtered = num_suggestions  # Keep all candidates if we already have very few
elif num_suggestions <= 10:
    num_suggestions_filtered = max(3, int(0.5 * num_suggestions))  # Keep 50% but at least 3
elif num_suggestions <= 20:
    num_suggestions_filtered = max(5, int(0.35 * num_suggestions))  # Keep 35% but at least 5
else:
    num_suggestions_filtered = max(8, int(0.2 * num_suggestions))  # Keep 20% but at least 8
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

"""
# Main Option 1: single score being dropped (ablation study)
print("Ablation study being conducted by dropping one score at each time.")
keys = ["bert", "cos", "lm", "freq", "sentence"]
opt_src = "caselaw_onlybert"
opt_log_csv = pd.read_csv(f"optimization_log_{opt_src}.csv")
opt_log_csv = opt_log_csv.sort_values(by="mean_harmonic_mean", ascending=False).reset_index(drop=True)
idx = 0
optimal_weights = opt_log_csv.loc[idx, ["weight_bert", "weight_cos", "weight_lm", "weight_freq", "weight_sentence"]].to_dict()
stats = opt_log_csv.loc[idx, ["mean_harmonic_mean", "std_harmonic_mean"]].to_dict()
bert_weight = optimal_weights["weight_bert"]
cos_weight = optimal_weights["weight_cos"]
lm_weight = optimal_weights["weight_lm"]
freq_weight = optimal_weights["weight_freq"]
sentence_weight = optimal_weights["weight_sentence"]
weights = [bert_weight, cos_weight, lm_weight, freq_weight, sentence_weight]
print("original weights")
print(weights)

for weight_idx in range(len(weights)):
    weights_abl = weights.copy()
    print("dropped_key")
    print(keys[weight_idx])
    weights_abl[weight_idx] = 0
    print("weights_abl")
    print(weights_abl)

    #all_suggested_words, all_ranked_dictionaries = run_substitution_ranking(all_suggestions, global_score_dict,
    #all_suggested_tokens, all_ranked_dictionaries = run_substitution_ranking(all_suggestions, global_score_dict,
    #                                                                        df_subtlex, complex_words, eng_words, glove, 
    #                                                                        weights)
    all_suggested_tokens, all_ranked_dictionaries = run_substitution_ranking(all_suggestion_scores,
                                                                             df_subtlex, complex_words, eng_words, glove, 
                                                                             weights_abl)
    
    #print("all_suggested_tokens")
    #print(all_suggested_tokens)
    #print("all_ranked_dictionaries")
    #print(all_ranked_dictionaries)
    all_simple_sentences = simplify_batched_text(all_original_texts, all_suggested_tokens)
    print("Simplification complete.")

    #print("all_original_texts")
    #print(all_original_texts)
    #print("all_simple_texts")
    #print(all_simple_sentences)

    simple_path = f"./ablation_output_data/test/uslt_noss_supreme_test_wo_{keys[weight_idx]}.txt"
    write_simplifications_to_file(all_simple_sentences, simple_path, complex_path)
    #mean_scores, std_scores = evaluate_w_kfold_cross_val(all_simple_sentences)
    #print("mean_scores")
    #print(mean_scores)
    #print("std_scores")
    #print(std_scores)
"""

# Main Option 2: manually entered weights
bert_weight = 0.2272155769544756
cos_weight = 0.13954983832511
lm_weight = 0.0424107596140726
freq_weight = 3.17378268510755e-05
sentence_weight = 0.5907920872794906
weights = [bert_weight, cos_weight, lm_weight, freq_weight, sentence_weight]

all_suggested_tokens, all_ranked_dictionaries = run_substitution_ranking(all_suggestion_scores,
                                                                         df_subtlex, complex_words, eng_words, glove, 
                                                                         weights)

print("all_suggested_tokens")
print(all_suggested_tokens)
print("all_ranked_dictionaries")
print(all_ranked_dictionaries)
all_simple_sentences = simplify_batched_text(all_original_texts, all_suggested_tokens)
print("Simplification complete.")
if data_split == "test":
    #simple_path = f"./output_data/test/trial/uslt_noss_supreme_test_trial.txt"
    simple_path = f"./qual_deneme/output_data/qual_deneme_output.txt"
else:
    #simple_path = f"./output_data/{data_split}/trial/supreme_{data_split}_uslt_noss_trial.txt"
    simple_path = f"./qual_deneme/output_data/qual_deneme_output.txt"

write_simplifications_to_file(all_simple_sentences, simple_path, complex_path)
#qual_deneme haricinde aşağıyı uncommentle
"""
mean_scores, std_scores = evaluate_w_kfold_cross_val(all_simple_sentences, original_sentences=all_original_texts)

#print("all_original_texts")
#print(all_original_texts)
#print("all_simple_texts")
#print(all_simple_sentences)
mean_scores, std_scores = evaluate_w_kfold_cross_val(all_simple_sentences, original_sentences=all_original_texts)
print("mean_scores")
print(mean_scores)
print("std_scores")
print(std_scores)
"""

"""
# Main Option 3: trials based on the weights found during hyperparameter optimization
opt_src = "caselaw_onlybert"
#opt_src = "caselaw_onlybertprec"
#opt_src = "caselaw_bertfkgl"
#opt_src = "supreme_onlybert"
#opt_src = "all_train_onlybert"
opt_log_csv = pd.read_csv(f"optimization_log_{opt_src}.csv")
opt_log_csv = opt_log_csv.sort_values(by="mean_harmonic_mean", ascending=False).reset_index(drop=True)
#opt_log_csv = pd.read_csv("optimization_log_supreme.csv")
#opt_log_csv = opt_log_csv.sort_values(by="mean_harmonic_mean", ascending=True).reset_index(drop=True)
for idx in range(10):
    optimal_weights = opt_log_csv.loc[idx, ["weight_bert", "weight_cos", "weight_lm", "weight_freq", "weight_sentence"]].to_dict()
    stats = opt_log_csv.loc[idx, ["mean_harmonic_mean", "std_harmonic_mean"]].to_dict()
    print(f"Trial {idx+1}: {optimal_weights}")
    print(f"Stats: ", stats)
    bert_weight = optimal_weights["weight_bert"]
    cos_weight = optimal_weights["weight_cos"]
    lm_weight = optimal_weights["weight_lm"]
    freq_weight = optimal_weights["weight_freq"]
    sentence_weight = optimal_weights["weight_sentence"]
    weights = [bert_weight, cos_weight, lm_weight, freq_weight, sentence_weight]

    all_suggested_tokens, all_ranked_dictionaries = run_substitution_ranking(all_suggestion_scores,
                                                                             df_subtlex, complex_words, eng_words, glove, 
                                                                             weights)
    all_simple_sentences = simplify_batched_text(all_original_texts, all_suggested_tokens)
    print("Simplification complete.")
    if data_split == "test":
        simple_path = f"./output_data/test/trial_{opt_src}/uslt_noss_supreme_test_trial{idx}.txt"
    else:
        simple_path = f"./output_data/{data_split}/trial_{opt_src}/supreme_{data_split}_uslt_noss_trial{idx}.txt"
    print("simple_path: ", simple_path)
    write_simplifications_to_file(all_simple_sentences, simple_path, complex_path)
    mean_scores, std_scores = evaluate_w_kfold_cross_val(all_simple_sentences, original_sentences=all_original_texts)
    print("mean_scores")
    print(mean_scores)
    print("std_scores")
    print(std_scores)
"""
