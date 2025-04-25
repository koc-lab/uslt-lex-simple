import os
import gc
import psutil
import optuna
import numpy as np
import pandas as pd
from easse.fkgl import corpus_fkgl
from readability import Readability
from bert_score import score as bertscore
from substitution_ranker import run_substitution_ranking
from simplify import simplify_batched_text

#LOG_FILE = "optimization_log_supreme_onlyfkgl.csv"
LOG_FILE = "optimization_log_caselaw_onlybert.csv"
#LOG_FILE = "optimization_log_caselaw_onlybertprec.csv"
#LOG_FILE = "optimization_log_caselaw_bertfkgl.csv"
#LOG_FILE = "optimization_log_supreme_onlybert.csv"
#LOG_FILE = "optimization_log_supreme_train_onlybert.csv"
#LOG_FILE = "optimization_log_all_train_onlybert.csv"

def log_memory_usage(trial):
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    print(f"Trial {trial.number}: Memory usage = {mem:.2f} MB")

def log_results(trial_number, weights, mean_harmonic_mean, std_harmonic_mean):
    """
    log_entry = {
        "trial_number": trial_number,
        "weight_bert": weights[0],
        "weight_freq": weights[1],
        "weight_lm": weights[2],
        "weight_cos": weights[3],
        "weight_len": weights[4],
        "mean_harmonic_mean": mean_harmonic_mean,
        "std_harmonic_mean": std_harmonic_mean
    }
    """
    log_entry = {
        "trial_number": trial_number,
        "weight_bert": weights[0],
        "weight_cos": weights[1],
        "weight_lm": weights[2],
        "weight_freq": weights[3],
        "weight_sentence": weights[4],
        "mean_harmonic_mean": mean_harmonic_mean,
        "std_harmonic_mean": std_harmonic_mean
    }

    # Check if the file exists, if not, create it with proper headers
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame([log_entry])
    else:
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)

    # Save back to CSV
    df.to_csv(LOG_FILE, index=False)
    print(f"Logged Trial {trial_number}: {log_entry}")

def evaluate_w_kfold_cross_val(simple_sentences, original_sentences=None, k=1, hyperparam_opt=False):
    #scores_array = np.zeros((3, k))
    k = 1
    scores_array = np.zeros((4, k))

    #fold_size = len(simple_sentences) // k
    fold_size = len(simple_sentences)
    for i in range(k):
        low, high = i * fold_size, (i + 1) * fold_size
        print("(low,high)")
        print((low,high))
        test_sentences = simple_sentences[low:high]
        test_org_sentences = original_sentences[low:high]

        fkgl = corpus_fkgl(test_sentences)
        #dc = Readability(' '.join(test_sentences)).dale_chall().score
        dc = Readability('\n'.join(test_sentences)).dale_chall().score
        prec, recall, f1 = bertscore(test_sentences, test_org_sentences, lang="en", rescale_with_baseline=True)
        #bert_score_final = f1.mean()
        bert_score_final = prec.mean()

        #harmonic_mean = 2 / (1 / (1-bert_score_final) + 1 / (fkgl/12))
        harmonic_mean = 2 / (1 / dc + 1 / fkgl)

        scores_array[0, i] = dc
        scores_array[1, i] = fkgl
        scores_array[2, i] = harmonic_mean
        scores_array[3, i] = bert_score_final

    mean_scores = np.mean(scores_array, axis=1)
    std_scores = np.std(scores_array, axis=1)
    print("mean_scores")
    print(mean_scores)
    if hyperparam_opt:
        #return mean_scores[2], std_scores[2]
        return mean_scores[3], std_scores[3]
        #return mean_scores[0], std_scores[0]
        #return mean_scores[1], std_scores[1]
    else:
        return mean_scores, std_scores

def optimize_hyperparameters(
    all_suggestion_scores, df_subtlex, complex_words, eng_words, 
    all_cleaned_texts, all_original_texts, model, tokenizer, 
    glove, device, n_trials=500):
    
    def objective(trial):
        #log_memory_usage(trial)
        trial_number = trial.number
        print(f"Trial {trial_number+1}/{n_trials}")

        # Sample weights from Dirichlet distribution to ensure sum = 1
        #raw_weights = np.array([-np.log(trial.suggest_float(f'raw_weight_{i}', 0.0, 1.0)) for i in range(5)])

        # Sample weights from uniform distribution and scale s.t. sum = 1
        raw_weights = np.array([trial.suggest_float(f'raw_weight_{i}', 0.0, 1.0) for i in range(5)])
        weights = raw_weights / np.sum(raw_weights)  # Normalize to sum = 1
        for i in range(5):
            trial.set_user_attr(f"weight_{i}", weights[i])

        #all_suggested_words, _ = run_substitution_ranking(
        all_suggested_tokens, _ = run_substitution_ranking(
            all_suggestion_scores, df_subtlex, complex_words, eng_words, 
            glove, weights
        )

        #all_simple_sentences = simplify_batched_text(all_original_texts, all_suggested_words)
        all_simple_sentences = simplify_batched_text(all_original_texts, all_suggested_tokens)
        mean_harmonic_mean, std_harmonic_mean = evaluate_w_kfold_cross_val(all_simple_sentences, all_original_texts, hyperparam_opt=True)

        del all_suggested_tokens, all_simple_sentences
        gc.collect()

        log_results(trial_number, weights, mean_harmonic_mean, std_harmonic_mean)
        return mean_harmonic_mean  # Optuna minimizes this

    # Run Bayesian Optimization
    study = optuna.create_study(direction="maximize") #maximize if bert_score, minimize if dc or fkgl or harmonic mean
    #study = optuna.create_study(direction="minimize") #maximize if bert_score, minimize if dc or fkgl or harmonic mean
    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)

    return study.best_params
