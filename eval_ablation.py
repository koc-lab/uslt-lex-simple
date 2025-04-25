## install easse and readability packages

from easse.sari import corpus_sari
from evaluate import load
from easse.fkgl import corpus_fkgl
import textstat
from bert_score import score as bertscore
from readability import Readability

import numpy as np
import pandas as pd
import random

input_file_og = open("lex-simple-500-dataset/test/supreme_org_test.txt","r").read().strip().split('\n')
ref_file1_og = open("lex-simple-500-dataset/test/supreme_test_ref1.txt","r").read().strip().split('\n')
ref_file2_og = open("lex-simple-500-dataset/test/supreme_test_ref2.txt","r").read().strip().split('\n')
ref_file3_og = open("lex-simple-500-dataset/test/supreme_test_ref3.txt","r").read().strip().split('\n')
uslt_noss_wo_bert_og = open("ablation_output_data/test/uslt_noss_supreme_test_wo_bert.txt","r").read().strip().split('\n')
uslt_noss_wo_cos_sim_og = open("ablation_output_data/test/uslt_noss_supreme_test_wo_cos.txt","r").read().strip().split('\n')
uslt_noss_wo_lm_loss_og = open("ablation_output_data/test/uslt_noss_supreme_test_wo_lm.txt","r").read().strip().split('\n')
uslt_noss_wo_freq_feat_og = open("ablation_output_data/test/uslt_noss_supreme_test_wo_freq.txt","r").read().strip().split('\n')
uslt_noss_wo_sentence_og = open("ablation_output_data/test/uslt_noss_supreme_test_wo_sentence.txt","r").read().strip().split('\n')
#uslt_zero_og = open("files/uslt_noss_sıfır_deneme_supreme_test.txt","r").read().strip().split('\n')
uslt_noss_og = open("output_data/test/uslt_noss_supreme_test.txt","r").read().strip().split('\n')
#uslt_noss_og = open("files/uslt_noss_yeni_deneme_supreme_test.txt","r").read().strip().split('\n')
#uslt_ss_og = open("files/uslt_supreme_test.txt","r").read().strip().split('\n')
uslt_ss_og = open("output_data/test/uslt_ss_supreme_test.txt","r").read().strip().split('\n')

sari = load("sari")

fold_size = 50
fold_count = 50//fold_size
num_metrics = 4
num_baselines = 7
scores_array = np.zeros((num_metrics,num_baselines,fold_count))
for i in range(fold_count):
    low = i*fold_size
    high = (i+1)*fold_size

    input_file = input_file_og[low:high]
    uslt_noss_wo_bert = uslt_noss_wo_bert_og[low:high]
    uslt_noss_wo_lm_loss = uslt_noss_wo_lm_loss_og[low:high]
    uslt_noss_wo_cos_sim = uslt_noss_wo_cos_sim_og[low:high]
    uslt_noss_wo_freq_feat = uslt_noss_wo_freq_feat_og[low:high]
    uslt_noss_wo_sentence = uslt_noss_wo_sentence_og[low:high]
    uslt_noss = uslt_noss_og[low:high]
    uslt_ss = uslt_ss_og[low:high]
    ref_file1 = ref_file1_og[low:high]
    ref_file2 = ref_file2_og[low:high]
    ref_file3 = ref_file3_og[low:high]

    input_dc = Readability(' '.join(input_file)).dale_chall().score
    uslt_noss_wo_bert_dc = Readability(' '.join(uslt_noss_wo_bert)).dale_chall().score
    uslt_noss_wo_lm_loss_dc = Readability(' '.join(uslt_noss_wo_lm_loss)).dale_chall().score
    uslt_noss_wo_cos_sim_dc = Readability(' '.join(uslt_noss_wo_cos_sim)).dale_chall().score
    uslt_noss_wo_freq_feat_dc = Readability(' '.join(uslt_noss_wo_freq_feat)).dale_chall().score
    uslt_noss_wo_sentence_dc = Readability(' '.join(uslt_noss_wo_sentence)).dale_chall().score
    uslt_noss_dc = Readability(' '.join(uslt_noss)).dale_chall().score
    uslt_ss_dc = Readability(' '.join(uslt_ss)).dale_chall().score

    uslt_noss_wo_bert_fkgl = corpus_fkgl(uslt_noss_wo_bert)
    uslt_noss_wo_lm_loss_fkgl = corpus_fkgl(uslt_noss_wo_lm_loss)
    uslt_noss_wo_cos_sim_fkgl = corpus_fkgl(uslt_noss_wo_cos_sim)
    uslt_noss_wo_freq_feat_fkgl = corpus_fkgl(uslt_noss_wo_freq_feat)
    uslt_noss_wo_sentence_fkgl = corpus_fkgl(uslt_noss_wo_sentence)
    uslt_noss_fkgl = corpus_fkgl(uslt_noss)
    uslt_ss_fkgl = corpus_fkgl(uslt_ss)

# muss_fkgl = Readability(' '.join(muss_test)).flesch_kincaid().score
# access_fkgl = Readability(' '.join(acces_test)).flesch_kincaid().score
# recls_fkgl = Readability(' '.join(recls_outputs)).flesch_kincaid().score
# lsbert_fkgl = Readability(' '.join(lsbert_outputs)).flesch_kincaid().score
# lsbert_ourcwi_fkgl = Readability(' '.join(lsbert_outputs_ourcwi)).flesch_kincaid().score
# tst_fkgl = Readability(' '.join(tst_outputs)).flesch_kincaid().score
# uslt_noss_fkgl = Readability(' '.join(uslt_noss)).flesch_kincaid().score
# uslt_fkgl = Readability(' '.join(uslt_ss)).flesch_kincaid().score

    """
    uslt_noss_wo_bert_sari = corpus_sari(orig_sents=input_file,  
                                        sys_sents=uslt_noss_wo_bert, 
                                        refs_sents=[ref_file1,
                                                    ref_file2,  
                                                    ref_file3])
    uslt_noss_wo_lm_loss_sari = corpus_sari(orig_sents=input_file,  
                                            sys_sents=uslt_noss_wo_lm_loss, 
                                            refs_sents=[ref_file1,
                                                        ref_file2,  
                                                        ref_file3])
    uslt_noss_wo_cos_sim_sari = corpus_sari(orig_sents=input_file,  
                                            sys_sents=uslt_noss_wo_cos_sim, 
                                            refs_sents=[ref_file1,
                                                        ref_file2,  
                                                        ref_file3])
    uslt_noss_wo_freq_feat_sari = corpus_sari(orig_sents=input_file,  
                                            sys_sents=uslt_noss_wo_freq_feat, 
                                            refs_sents=[ref_file1,
                                                        ref_file2,  
                                                        ref_file3])
    uslt_noss_wo_sentence_sari = corpus_sari(orig_sents=input_file,  
                                                sys_sents=uslt_noss_wo_sentence, 
                                                refs_sents=[ref_file1,
                                                            ref_file2,  
                                                            ref_file3])
    uslt_noss_sari = corpus_sari(orig_sents=input_file,  
                                sys_sents=uslt_noss, 
                                refs_sents=[ref_file1,
                                            ref_file2,  
                                            ref_file3])
    uslt_ss_sari = corpus_sari(orig_sents=input_file,  
                            sys_sents=uslt_ss, 
                            refs_sents=[ref_file1,
                                        ref_file2,  
                                        ref_file3])
    """

    refs_concatenated = []
    ref_count = 3
    for l in range(len(ref_file1)):
        refs_concatenated.append([ref_file1[l], ref_file2[l], ref_file3[l]])       
    uslt_noss_wo_bert_sari = sari.compute(sources=input_file, predictions=uslt_noss_wo_bert, references=refs_concatenated)["sari"]
    uslt_noss_wo_lm_loss_sari = sari.compute(sources=input_file, predictions=uslt_noss_wo_lm_loss, references=refs_concatenated)["sari"]
    uslt_noss_wo_cos_sim_sari = sari.compute(sources=input_file, predictions=uslt_noss_wo_cos_sim, references=refs_concatenated)["sari"]
    uslt_noss_wo_freq_feat_sari = sari.compute(sources=input_file, predictions=uslt_noss_wo_freq_feat, references=refs_concatenated)["sari"]
    uslt_noss_wo_sentence_sari = sari.compute(sources=input_file, predictions=uslt_noss_wo_sentence, references=refs_concatenated)["sari"]
    uslt_noss_sari = sari.compute(sources=input_file, predictions=uslt_noss, references=refs_concatenated)["sari"]
    uslt_ss_sari = sari.compute(sources=input_file, predictions=uslt_ss, references=refs_concatenated)["sari"]
    
    uslt_noss_wo_bert_bertsc = bertscore(uslt_noss_wo_bert, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    uslt_noss_wo_lm_loss_bertsc = bertscore(uslt_noss_wo_lm_loss, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    uslt_noss_wo_cos_sim_bertsc = bertscore(uslt_noss_wo_cos_sim, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    uslt_noss_wo_freq_feat_bertsc = bertscore(uslt_noss_wo_freq_feat, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    uslt_noss_wo_sentence_bertsc = bertscore(uslt_noss_wo_sentence, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    uslt_noss_bertsc = bertscore(uslt_noss, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    uslt_ss_bertsc = bertscore(uslt_ss, input_file, lang="en", rescale_with_baseline=True)[2].mean()

    score_dict = {"uslt_noss_wo_bert":[uslt_noss_wo_bert_sari,uslt_noss_wo_bert_fkgl,uslt_noss_wo_bert_dc,uslt_noss_wo_bert_bertsc], 
                   "uslt_noss_wo_lm_loss":[uslt_noss_wo_lm_loss_sari,uslt_noss_wo_lm_loss_fkgl,uslt_noss_wo_lm_loss_dc,uslt_noss_wo_lm_loss_bertsc], 
                   "uslt_noss_wo_cos_sim":[uslt_noss_wo_cos_sim_sari,uslt_noss_wo_cos_sim_fkgl,uslt_noss_wo_cos_sim_dc,uslt_noss_wo_cos_sim_bertsc], 
                   "uslt_noss_wo_freq_feat":[uslt_noss_wo_freq_feat_sari,uslt_noss_wo_freq_feat_fkgl,uslt_noss_wo_freq_feat_dc,uslt_noss_wo_freq_feat_bertsc], 
                   "uslt_noss_wo_sentence":[uslt_noss_wo_sentence_sari,uslt_noss_wo_sentence_fkgl,uslt_noss_wo_sentence_dc,uslt_noss_wo_sentence_bertsc],
                   "uslt no ss":[uslt_noss_sari,uslt_noss_fkgl,uslt_noss_dc,uslt_noss_bertsc], 
                   "uslt ss":[uslt_ss_sari,uslt_ss_fkgl,uslt_ss_dc,uslt_ss_bertsc]}
    c = 0
    for key in score_dict:
        for metric in range(num_metrics):
            print("key: ", key)
            print("metric: ", metric)
            print(score_dict[key][metric])
            scores_array[metric,c,i] = score_dict[key][metric]
        c += 1
    
final_score_dict = np.mean(scores_array,axis=2)
df_means = pd.DataFrame(final_score_dict,index=['SARI', 'FKGL','DC','BERTScore'],columns=['uslt_noss_wo_bert','uslt_noss_wo_lm_loss','uslt_noss_wo_cos_sim','uslt_noss_wo_freq_feat','uslt_noss_wo_sentence','uslt no ss','uslt ss'])
print(df_means)
stds = np.std(scores_array,axis=2)
df_stds = pd.DataFrame(stds,index=['SARI', 'FKGL','DC','BERTScore'],columns=['uslt_noss_wo_bert','uslt_noss_wo_lm_loss','uslt_noss_wo_cos_sim','uslt_noss_wo_freq_feat','uslt_noss_wo_sentence','uslt no ss','uslt ss'])
print(df_stds)
