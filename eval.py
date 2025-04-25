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

# test set paths
input_file_og = open("lex-simple-dataset/test/supreme_org_test.txt","r").read().strip().split('\n')
ref_file1_og = open("lex-simple-dataset/test/supreme_test_ref1.txt","r").read().strip().split('\n')
ref_file2_og = open("lex-simple-dataset/test/supreme_test_ref2.txt","r").read().strip().split('\n')
ref_file3_og = open("lex-simple-dataset/test/supreme_test_ref3.txt","r").read().strip().split('\n')
muss_og = open("output_data/test/muss_supreme_test.txt","r").read().strip().split('\n')
access_og = open("output_data/test/access_supreme_test.txt","r").read().strip().split('\n')
recls_outputs_og = open("output_data/test/recls_supreme_test.txt","r").read().strip().split('\n')
lsbert_outputs_og = open("output_data/test/lsbert_supreme_test.txt","r").read().strip().split('\n')
lsbert_outputs_ourcwi_og = open("output_data/test/lsbert_ourcwi_supreme_test.txt","r").read().strip().split('\n')
tst_outputs_og = open("output_data/test/gector_supreme_test.txt","r").read().strip().split("\n")
uslt_og = open("output_data/test/uslt_noss_supreme_test.txt","r").read().strip().split('\n')
#uslt_og = open("output_data/test/trial_caselaw_onlybert/uslt_noss_supreme_test_trial0.txt","r").read().strip().split('\n')
uslt_ss_og = open("output_data/test/uslt_ss_supreme_test.txt","r").read().strip().split('\n')

# val set paths
#input_file_og = open("lex-simple-dataset/val/supreme_org_val.txt","r").read().strip().split('\n')
#ref_file1_og = open("lex-simple-dataset/val/supreme_val_ref1.txt","r").read().strip().split('\n')
#ref_file2_og = open("lex-simple-dataset/val/supreme_val_ref2.txt","r").read().strip().split('\n')
#ref_file3_og = open("lex-simple-dataset/val/supreme_val_ref3.txt","r").read().strip().split('\n')
#muss_og = open("output_data/val/supreme_val_muss.txt","r").read().strip().split('\n')
#access_og = open("output_data/val/supreme_val_access.txt","r").read().strip().split('\n')
#recls_outputs_og = open("output_data/val/supreme_val_recls.txt","r").read().strip().split('\n')
#lsbert_outputs_og = open("output_data/val/supreme_val_lsbert.txt","r").read().strip().split('\n')
#lsbert_outputs_ourcwi_og = open("output_data/val/supreme_val_lsbert_ourcwi.txt","r").read().strip().split('\n')
#tst_outputs_og = open("output_data/val/supreme_val_gector.txt","r").read().strip().split("\n")
#uslt_og = open("output_data/val/supreme_val_uslt_noss.txt","r").read().strip().split('\n')
#uslt_ss_og = open("output_data/val/supreme_val_uslt_ss.txt","r").read().strip().split('\n')

sari = load("sari")

fold_size = 50
fold_count = 50//fold_size
num_metrics = 4
num_baselines = 12
scores_array = np.zeros((num_metrics,num_baselines,fold_count))
for i in range(fold_count):
    low = i*fold_size
    high = (i+1)*fold_size

    input_file = input_file_og[low:high]
    muss_test = muss_og[low:high]
    access_test = access_og[low:high]
    recls_outputs= recls_outputs_og[low:high]
    lsbert_outputs = lsbert_outputs_og[low:high]
    lsbert_outputs_ourcwi = lsbert_outputs_ourcwi_og[low:high]
    tst_outputs = tst_outputs_og[low:high]
    uslt = uslt_og[low:high]
    uslt_ss = uslt_ss_og[low:high]
    ref_file1 = ref_file1_og[low:high]
    ref_file2 = ref_file2_og[low:high]
    ref_file3 = ref_file3_og[low:high]

    input_dc = Readability(' '.join(input_file)).dale_chall().score
    muss_dc = Readability(' '.join(muss_test)).dale_chall().score
    access_dc = Readability(' '.join(access_test)).dale_chall().score
    recls_dc = Readability(' '.join(recls_outputs)).dale_chall().score
    lsbert_dc = Readability(' '.join(lsbert_outputs)).dale_chall().score
    lsbert_ourcwi_dc = Readability(' '.join(lsbert_outputs_ourcwi)).dale_chall().score
    tst_dc = Readability(' '.join(tst_outputs)).dale_chall().score
    uslt_ss_dc = Readability(' '.join(uslt_ss)).dale_chall().score
    uslt_dc = Readability(' '.join(uslt)).dale_chall().score
    ref1_dc = Readability(' '.join(ref_file1)).dale_chall().score
    ref2_dc = Readability(' '.join(ref_file2)).dale_chall().score
    ref3_dc = Readability(' '.join(ref_file3)).dale_chall().score

    muss_fkgl = corpus_fkgl(muss_test)
    access_fkgl = corpus_fkgl(access_test)
    recls_fkgl = corpus_fkgl(recls_outputs)
    lsbert_fkgl = corpus_fkgl(lsbert_outputs)
    lsbert_ourcwi_fkgl = corpus_fkgl(lsbert_outputs_ourcwi)
    tst_fkgl = corpus_fkgl(tst_outputs)
    uslt_ss_fkgl = corpus_fkgl(uslt_ss)
    uslt_fkgl = corpus_fkgl(uslt)
    ref1_fkgl = corpus_fkgl(ref_file1)
    ref2_fkgl = corpus_fkgl(ref_file2)
    ref3_fkgl = corpus_fkgl(ref_file3)

    """
    muss_fkgl, access_fkgl, recls_fkgl, lsbert_fkgl, lsbert_ourcwi_fkgl, tst_fkgl, uslt_ss_fkgl, uslt_fkgl, ref1_fkgl, ref2_fkgl, ref3_fkgl = 0,0,0,0,0,0,0,0,0,0,0
    for s in range(len(input_file)):
        muss_fkgl += textstat.flesch_kincaid_grade(muss_test[s])/len(input_file)
        access_fkgl += textstat.flesch_kincaid_grade(access_test[s])/len(input_file)
        recls_fkgl += textstat.flesch_kincaid_grade(recls_outputs[s])/len(input_file)
        lsbert_fkgl += textstat.flesch_kincaid_grade(lsbert_outputs[s])/len(input_file)
        lsbert_ourcwi_fkgl += textstat.flesch_kincaid_grade(lsbert_outputs_ourcwi[s])/len(input_file)
        tst_fkgl += textstat.flesch_kincaid_grade(tst_outputs[s])/len(input_file)
        uslt_ss_fkgl += textstat.flesch_kincaid_grade(uslt_ss[s])/len(input_file)
        uslt_fkgl += textstat.flesch_kincaid_grade(uslt[s])/len(input_file)
        ref1_fkgl += textstat.flesch_kincaid_grade(ref_file1[s])/len(input_file)
        ref2_fkgl += textstat.flesch_kincaid_grade(ref_file2[s])/len(input_file)
        ref3_fkgl += textstat.flesch_kincaid_grade(ref_file3[s])/len(input_file)
    """

    """
    muss_fkgl = Readability(' '.join(muss_test)).flesch_kincaid().score
    access_fkgl = Readability(' '.join(access_test)).flesch_kincaid().score
    recls_fkgl = Readability(' '.join(recls_outputs)).flesch_kincaid().score
    lsbert_fkgl = Readability(' '.join(lsbert_outputs)).flesch_kincaid().score
    lsbert_ourcwi_fkgl = Readability(' '.join(lsbert_outputs_ourcwi)).flesch_kincaid().score
    tst_fkgl = Readability(' '.join(tst_outputs)).flesch_kincaid().score
    uslt_ss_fkgl = Readability(' '.join(uslt_ss)).flesch_kincaid().score
    uslt_fkgl = Readability(' '.join(uslt)).flesch_kincaid().score
    ref1_fkgl = Readability(' '.join(ref_file1)).flesch_kincaid().score
    ref2_fkgl = Readability(' '.join(ref_file2)).flesch_kincaid().score
    ref3_fkgl = Readability(' '.join(ref_file3)).flesch_kincaid().score
    """
    
    """
    muss_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=muss_test, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    access_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=access_test, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    recls_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=recls_outputs, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    lsbert_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=lsbert_outputs, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    lsbert_ourcwi_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=lsbert_outputs_ourcwi, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    tst_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=tst_outputs, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    uslt_ss_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=uslt_ss, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    uslt_sari = corpus_sari(orig_sents=input_file,  
                sys_sents=uslt, 
                refs_sents=[ref_file1,
                            ref_file2,  
                            ref_file3])
    ref1_sari = corpus_sari(orig_sents=input_file,
                sys_sents=ref_file1,
                refs_sents=[ref_file2,
                            ref_file3])
    ref2_sari = corpus_sari(orig_sents=input_file,
                sys_sents=ref_file2,
                refs_sents=[ref_file1,
                            ref_file3])
    ref3_sari = corpus_sari(orig_sents=input_file,
                sys_sents=ref_file3,
                refs_sents=[ref_file1,
                            ref_file2])
    """
    
    refs_concatenated = []
    ref_count = 3
    for l in range(len(ref_file1)):
        refs_concatenated.append([ref_file1[l], ref_file2[l], ref_file3[l]])       
    muss_sari = sari.compute(sources=input_file, predictions=muss_test, references=refs_concatenated)["sari"]
    access_sari = sari.compute(sources=input_file, predictions=access_test, references=refs_concatenated)["sari"]
    recls_sari = sari.compute(sources=input_file, predictions=recls_outputs, references=refs_concatenated)["sari"]
    lsbert_sari = sari.compute(sources=input_file, predictions=lsbert_outputs, references=refs_concatenated)["sari"]
    lsbert_ourcwi_sari = sari.compute(sources=input_file, predictions=lsbert_outputs_ourcwi, references=refs_concatenated)["sari"]
    tst_sari = sari.compute(sources=input_file, predictions=tst_outputs, references=refs_concatenated)["sari"]
    uslt_ss_sari = sari.compute(sources=input_file, predictions=uslt_ss, references=refs_concatenated)["sari"]
    uslt_sari = sari.compute(sources=input_file, predictions=uslt, references=refs_concatenated)["sari"]
    ref1_sari = sari.compute(sources=input_file, predictions=ref_file1, references=refs_concatenated)["sari"]
    ref2_sari = sari.compute(sources=input_file, predictions=ref_file2, references=refs_concatenated)["sari"]
    ref3_sari = sari.compute(sources=input_file, predictions=ref_file3, references=refs_concatenated)["sari"]    
    
    muss_bert = bertscore(muss_test, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    access_bert = bertscore(access_test, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    recls_bert = bertscore(recls_outputs, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    lsbert_bert = bertscore(lsbert_outputs, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    lsbert_ourcwi_bert = bertscore(lsbert_outputs_ourcwi, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    tst_bert = bertscore(tst_outputs, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    uslt_ss_bert = bertscore(uslt_ss, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    uslt_bert = bertscore(uslt, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    ref1_bert = bertscore(ref_file1, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    ref2_bert = bertscore(ref_file2, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    ref3_bert = bertscore(ref_file3, input_file, lang="en", rescale_with_baseline=True)[2].mean()
    
    """
    ref1_add_ref = random.choice([ref_file2, ref_file3])
    ref2_add_ref = random.choice([ref_file1, ref_file3])
    ref3_add_ref = random.choice([ref_file1, ref_file2])
    
    ref1_sari = corpus_sari(orig_sents=input_file,
                sys_sents=ref_file1,
                refs_sents=[ref_file2,
                            ref_file3,
                            ref1_add_ref])
    ref2_sari = corpus_sari(orig_sents=input_file,
                sys_sents=ref_file2,
                refs_sents=[ref_file1,
                            ref_file3,
                            ref2_add_ref])
    ref3_sari = corpus_sari(orig_sents=input_file,
                sys_sents=ref_file3,
                refs_sents=[ref_file1,
                            ref_file2,
                            ref3_add_ref])
    """
        
    #gold_ref_avg_dc = np.mean([ref1_dc, ref2_dc, ref3_dc])
    #gold_ref_avg_fkgl = np.mean([ref1_fkgl, ref2_fkgl, ref3_fkgl])
    #gold_ref_avg_sari = np.mean([ref1_sari, ref2_sari, ref3_sari])
    gold_ref_avg_dc = np.mean([ref1_dc, ref2_dc, ref3_dc])
    gold_ref_avg_fkgl = np.mean([ref1_fkgl, ref2_fkgl, ref3_fkgl])
    gold_ref_avg_sari = np.mean([ref1_sari, ref2_sari, ref3_sari])
    gold_ref_avg_bert = np.mean([ref1_bert, ref2_bert, ref3_bert])

    score_dict = {"access":[access_sari,access_fkgl,access_dc,access_bert], 
                   "muss":[muss_sari,muss_fkgl,muss_dc, muss_bert], 
                   "recls":[recls_sari,recls_fkgl,recls_dc, recls_bert], 
                   "lsbert":[lsbert_sari,lsbert_fkgl,lsbert_dc, lsbert_bert], 
                   "lsbert_ourcwi":[lsbert_ourcwi_sari,lsbert_ourcwi_fkgl,lsbert_ourcwi_dc, lsbert_ourcwi_bert],
                   "tst":[tst_sari, tst_fkgl, tst_dc, tst_bert], 
                   "uslt ss":[uslt_ss_sari, uslt_ss_fkgl, uslt_ss_dc, uslt_ss_bert],
                   "uslt noss":[uslt_sari,uslt_fkgl,uslt_dc, uslt_bert], 
                   "ref1": [ref1_sari,ref1_fkgl,ref1_dc,ref1_bert],
                   "ref2": [ref2_sari,ref2_fkgl,ref2_dc,ref2_bert],
                   "ref3": [ref3_sari,ref3_fkgl,ref3_dc,ref3_bert],
                   "gold_ref_avg": [gold_ref_avg_sari, gold_ref_avg_fkgl, gold_ref_avg_dc, gold_ref_avg_bert]}

    c = 0
    for key in score_dict:
        for metric in range(num_metrics):
            scores_array[metric,c,i] = score_dict[key][metric]
        c += 1
    
final_score_dict = np.mean(scores_array,axis=2)
df_means = pd.DataFrame(final_score_dict,index=['SARI', 'FKGL','DC', 'BERTScore'],columns=['access','muss','recls','lsbert','lsbert_ourcwi','tst','uslt ss','uslt noss','ref1','ref2','ref3','gold ref avg'])
print(df_means)
stds = np.std(scores_array,axis=2)
df_stds = pd.DataFrame(stds,index=['SARI', 'FKGL','DC', 'BERTScore'],columns=['access','muss','recls','lsbert','lsbert_ourcwi','tst','uslt ss','uslt noss','ref1','ref2','ref3','gold ref avg'])
print(df_stds)
