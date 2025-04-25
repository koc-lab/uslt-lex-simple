import copy
import re
import language_tool_python
from construct_masked_lm import process_cwi
from suggestion_generator import suggestion_generator
#from substitution_ranker import substitution_ranker
from utils import true_correct, sentence_builder, sentence_builder_word

def simplify_batched_text(all_original_texts, all_suggested_tokens):
    lang_tool = language_tool_python.LanguageTool('en-US')
    all_simple_sentences = []
    sentence_count = 0
    print("\n")
    for original_text, suggested_words in zip(all_original_texts, all_suggested_tokens):
        sentence_count += 1
        print(f"Sentence {sentence_count}/{len(all_original_texts)}", end="\r")
        #produced_text = sentence_builder(original_text, suggested_words, tokenizer)
        final_text = sentence_builder(original_text, suggested_words)
        all_simple_sentences.append(final_text.strip())
    print("\n")
    lang_tool.close()
    del lang_tool
    return all_simple_sentences


def simplify_batched_text_words(all_original_texts, all_suggested_words):
    lang_tool = language_tool_python.LanguageTool('en-US')
    all_simple_sentences = []
    sentence_count = 0
    print("\n")
    for original_text, suggested_words in zip(all_original_texts, all_suggested_words):
        sentence_count += 1
        print(f"Sentence {sentence_count}/{len(all_original_texts)}", end="\r")
        produced_text = sentence_builder_word(original_text, suggested_words)
        try:
            words = suggested_words.values()
        except:
            words = None
        #final_text = true_correct(produced_text, words)
        final_text = true_correct(produced_text, words, lang_tool)
        all_simple_sentences.append(final_text.strip())
    print("\n")
    lang_tool.close()
    del lang_tool
    return all_simple_sentences

"""
def simplify_text(original_text, tokenizer, model, 
                  df_subtlex, complex_words, eng_words, glove,
                  weight_bert,weight_cos,weight_lm,weight_freq,weight_len, 
                  num_suggestions,
                  ner_checker, nlp_ner, 
                  device):
    cwi_result = process_cwi(original_text, tokenizer, complex_words, ner_checker, nlp_ner)
    words_found = cwi_result['words_found']
    masked_inputs = cwi_result["masked_inputs"]
    cleaned_text = cwi_result["cleaned_text"]
    suggestion_dict = suggestion_generator(model, tokenizer, masked_inputs, words_found, num_suggestions, device)
    suggested_words, ranked_dictionary = substitution_ranker(suggestion_dict, df_subtlex, complex_words, eng_words, cleaned_text, 
                                                             model, tokenizer, glove, 
                                                             weight_bert, weight_cos, weight_lm, weight_freq, weight_len,
                                                             device)
    print("suggested_words")
    print(suggested_words)
    print("ranked_dictionary")
    print(ranked_dictionary)
    print("original_text")
    print(original_text)
    produced_text = sentence_builder(original_text, suggested_words)
    print("produced_text")
    print(produced_text)
    return [words_found, ranked_dictionary, suggested_words, produced_text]
"""

def simplify_document(input_path, tokenizer, model,
                      df_subtlex, complex_words, eng_words, glove,
                      weight_bert, weight_cos, weight_lm, weight_freq, weight_len,
                      num_suggestions, 
                      ner_checker, nlp_ner,
                      device):
    simple_lines=[]
    #with open(input_path,'r',encoding = "latin", errors="ignore") as f:
    with open(input_path, 'r', encoding = "utf-8", errors="ignore") as f:
        #for line in f:
            line = f.readlines()[0]
            original_text = line
            replaced_words, ranked_dictionary, suggested_words, produced_text = simplify_text(original_text, tokenizer, model, 
                                                                                              df_subtlex, complex_words, eng_words, glove, 
                                                                                              weight_bert, weight_cos, weight_lm, weight_freq, weight_len,
                                                                                              num_suggestions,
                                                                                              ner_checker, nlp_ner,
                                                                                              device)
            #new_text = line
            try:
                words = suggested_words.values()
            except:
                words = None
            final_text = true_correct(produced_text, words) # correcting grammatical issues to produce the final version
            simple_lines.append(final_text.strip())
    return simple_lines

def write_simplifications_to_file(simple_lines, simple_path, complex_path):
    complex_file = open(complex_path, "r", encoding="utf-8", errors="ignore")
    complex_lines = complex_file.read().split("\n")
    simple_file = open(simple_path, "w", encoding="utf-8", errors="ignore")
    for i in range(len(simple_lines)):
        try:
            simple_file.write(simple_lines[i].strip())
        except:
            simple_file.write(complex_lines[i].strip())
            print(f"problem with {i}")
        if i != len(simple_lines)-1:
            simple_file.write("\n")
    simple_file.close()
    complex_file.close()
