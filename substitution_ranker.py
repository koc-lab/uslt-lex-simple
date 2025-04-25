import numpy as np
import pandas as pd
import torchtext
from utils import check_pos, check_pos_bert, normalize_score, compute_global_stats, place_suggestion_into_sentence_token, place_suggestion_into_sentence_word
from ranking_scores import compute_frequency_score, compute_length_score, compute_cosine_similarity, compute_lm_perplexity_loss, compute_contextual_fit_score, edit_distance_score, bert_word_similarity_score, bert_sentence_similarity, compute_contextual_word_similarity

def score_substitutions(all_suggestions, 
                        df_subtlex, complex_words, eng_words, 
                        all_cleaned_texts,
                        model, sentence_model, tokenizer, glove,
                        device):
    """
    Runs substitution_ranker on each batch individually and collects results.
    
    Args:
        all_suggestions (list of dict): List where each element is a dictionary of suggestions for a batch.
        all_cleaned_texts (list of str): List of cleaned texts, one per batch.

    Returns:
        all_suggested_words (list of dict): List of best substitutions per batch.
        all_ranked_dictionaries (list of dict): List of ranked suggestions per batch.
    """

    # Step 1: Collect all scores across all sentences before normalization
    global_score_dict = {
        "bert": [],
        "freq": [],
        "len": [],
        "cos": [],
        "lm": [],
        "word": [],
        "sentence": []
    }

    global_stats_dict = {
        "bert": [],
        "freq": [],
        "len": [],
        "cos": [],
        "lm": [],
        "word": [],
        "sentence": []
    }

    all_results = []
    all_key_scores = []
    word_count = 0

    print("Scores for suggestions being calculated...")

    print("all_suggestions")
    print(all_suggestions)

    for batch_idx, suggestion_dict in enumerate(all_suggestions):
        print(f"Suggestion {batch_idx+1}/{len(all_suggestions)}", end="\r")

        original_text = all_cleaned_texts[batch_idx]

        # Compute raw scores without normalization
        scores, key_scores, word_count = collect_scores(
            suggestion_dictionary=suggestion_dict,
            df_subtlex=df_subtlex,
            complex_words=complex_words,
            eng_words=eng_words,
            original_text=original_text,
            model=model,
            sentence_model=sentence_model,
            tokenizer=tokenizer,
            glove=glove,
            device=device,
            word_count=word_count
        )

        # Collect scores for global normalization
        for key in global_score_dict.keys():
            global_score_dict[key].extend(scores[key])

        # Store batch results for later normalization
        all_results.append((suggestion_dict, scores, key_scores))
        all_key_scores.append(key_scores)

    print("\n")

    # Debug: Print statistics before normalization
    print("\n--- Raw Score Statistics Before Normalization ---")
    for key in global_score_dict.keys():
        print(f"{key} Score Statistics:")
        print(pd.DataFrame(global_score_dict[key]).describe())

    all_key_scores_normalized = []

    for key in global_stats_dict.keys():
        global_stats_dict[key] = compute_global_stats(global_scores=np.array(global_score_dict[key]))

    # Step 2: Normalize scores for each complex word candidate list
    for scores_dict in all_key_scores:
        key_score_normalized = {}
        for word_key in scores_dict.keys():
            word_score_dict = scores_dict[word_key]
            word_score_dict_normal = {}
            for candidate_key in word_score_dict.keys():
                candidate_score_dict = word_score_dict[candidate_key]
                candidate_score_dict_normal = {}
                for score_key in candidate_score_dict.keys():
                    #candidate_score_dict[score_key] = normalize_score(candidate_score_dict[score_key], score_type=score_key, global_scores=np.array(global_score_dict[score_key]))
                    candidate_score_dict_normal[score_key] = normalize_score(candidate_score_dict[score_key], score_type=score_key, global_stats=global_stats_dict[score_key])
                word_score_dict_normal[candidate_key] = candidate_score_dict_normal
            key_score_normalized[word_key] = word_score_dict_normal
        all_key_scores_normalized.append(key_score_normalized)
    
    #print("all_key_scores_normalized[0]")
    #print(all_key_scores_normalized[0])

    # Step 2.5: Normalize scores globally
    for key in global_score_dict.keys():
        global_score_dict[key] = np.array(global_score_dict[key])
        global_score_dict[key] = normalize_score(global_score_dict[key], score_type=key)

    # Debug: Print statistics after normalization
    print("\n--- Normalized Score Statistics ---")
    for key in global_score_dict.keys():
        print(f"{key} Normalized Score Statistics:")
        print(pd.DataFrame(global_score_dict[key]).describe())

    return all_key_scores_normalized

def collect_scores(suggestion_dictionary, df_subtlex, complex_words, eng_words, original_text, 
                                       model, sentence_model, tokenizer, glove, device, word_count):
    """
    Collects scores without normalization.
    """
    # Storage for raw scores before normalization
    key_score_list_dict = {}
    global_score_dict = {"bert": [], "freq": [], "len": [], "cos": [], "lm": [], "word": [], "sentence": []}

    # Compute scores for all suggestions
    for key, suggestions in suggestion_dictionary.items():
        candidate_parallel_list = []
        candidate_sentence_list = []
        candidate_score_list_dict = {}
        for candidate, bert_score in suggestions:
            score_list_dict = {}
            if candidate.startswith("##"):
                if not(key.startswith("##")):
                    continue
                else:
                    candidate_clean = candidate[2:]
                    key_clean = key[2:]
            else:
                #if key.startswith("##"):
                #    continue
                candidate_clean = candidate
                key_clean = key
            
            word_count += 1
            print("Total candidate count: ", word_count, end="\r")

            target_word_wo_occ = key.split("_[")[0] if "_[" in key else key
            occurrence_count = int(key.split("_[")[1][:-1]) if "_[" in key else 1
            target_word_clean = key_clean.split("_[")[0] if "_[" in key_clean else key_clean
            target_word = target_word_wo_occ.split()[0] if len(target_word_wo_occ.split()) > 1 else target_word_wo_occ

            candidate_parallel_list.append(candidate)
            #candidate_sentence, _ = place_suggestion_into_sentence_token(original_text, target_word, candidate, tokenizer, occurrence_count)
            candidate_sentence, _ = place_suggestion_into_sentence_word(original_text, target_word, candidate, occurrence_count)
            candidate_sentence_list.append(candidate_sentence)

            # Compute individual scores
            lm_score = compute_lm_perplexity_loss(original_text, target_word, candidate, model, tokenizer, target_occurrence_count=occurrence_count, device=device) #uses original key for BERT tokenizer compatibility
            #lm_score = compute_contextual_word_similarity(original_text, target_word, candidate, model, tokenizer, target_occurrence_count=occurrence_count, device=device) #uses original key for BERT tokenizer compatibility
            context_word_score = compute_contextual_word_similarity(original_text, target_word, candidate, model, tokenizer, target_occurrence_count=occurrence_count, device=device) #uses original key for BERT tokenizer compatibility
            freq_score = compute_frequency_score(candidate_clean, df_subtlex)
            #freq_score = bert_word_similarity_score(target_word_clean, candidate_clean, model, tokenizer, device)
            len_score = compute_length_score(candidate_clean)
            #len_score = edit_distance_score(target_word_clean, candidate_clean) #levenstein score
            #cos_score = compute_cosine_similarity(target_word_clean, candidate_clean, glove) #uses key_clean
            cos_score = compute_cosine_similarity(target_word_clean, candidate_clean, glove, tokenizer) #uses key_clean
            #cos_score = bert_sentence_similarity(target_word_clean, candidate_clean, sentence_model)
            #cos_score = word_movers_distance(target_word_clean, candidate_clean, word_vectors) #word movers distance score
            #context_score = compute_contextual_fit_score(original_text, target_word, candidate, model, tokenizer, target_occurence_count=occurrence_count)
            #cos_score = 

            # Check part-of-speech compatibility

            # Store raw scores
            score_list_dict["bert"] = bert_score
            score_list_dict["freq"] = freq_score
            score_list_dict["len"] = len_score
            score_list_dict["cos"] = cos_score
            score_list_dict["lm"] = lm_score
            score_list_dict["word"] = context_word_score
            global_score_dict["bert"].append(bert_score)
            global_score_dict["freq"].append(freq_score)
            global_score_dict["len"].append(len_score)
            global_score_dict["cos"].append(cos_score)
            global_score_dict["lm"].append(lm_score)
            global_score_dict["word"].append(context_word_score)
            candidate_score_list_dict[candidate] = score_list_dict
        
        if suggestion_dictionary[key] != []:
            sentence_similarity = bert_sentence_similarity([original_text], candidate_sentence_list, sentence_model)
            for i in range(len(candidate_parallel_list)):
                if type(sentence_similarity) == list:
                    sim = sentence_similarity[i]
                else:
                    sim = sentence_similarity
                candidate = candidate_parallel_list[i]
                global_score_dict["sentence"].append(sim)
                candidate_score_list_dict[candidate]["sentence"] = sim
        else:
            for i in range(len(candidate_parallel_list)):
                candidate = candidate_parallel_list[i]
                candidate_score_list_dict[candidate]["sentence"] = []
        
        #cos_score = bert_sentence_similarity([original_text], candidate_parallel_list, sentence_model)
        #score_list_dict["freq"].extend(sentence_similarity)
        #word_similarity = bert_sentence_similarity([target_word_clean], candidate_parallel_list, sentence_model)
        #score_list_dict["freq"].extend(word_similarity)
        key_score_list_dict[key] = candidate_score_list_dict
    return global_score_dict, key_score_list_dict, word_count

def rank_substitution_scores(suggestion_score_dict, df_subtlex, complex_words, eng_words, glove, weights, index):
    """
    Finalizes the scores and ranks the suggestions based on normalized scores.
    """
    #weight_bert, weight_cos, weight_lm, weight_freq, weight_len = weights
    #weight_bert, weight_cos, weight_lm, weight_freq, weight_len, weight_word, weight_sentence = weights
    weight_bert, weight_cos, weight_lm, weight_freq, weight_sentence = weights
    weight_len, weight_word = 0.0, 0.0

    #suggested_word = {}
    suggested_token = {}
    normalized_ranked_dictionary = {}

    #print("suggestion_score_dict")
    #print(suggestion_score_dict)

    for complex_word_key, candidate_score_dict in suggestion_score_dict.items():
        score_list = []
        #print("suggestion_score_dict[complex_word_key]")
        #print(suggestion_score_dict[complex_word_key])
        if suggestion_score_dict[complex_word_key] == {}:
            score_list.append((complex_word_key, 0.0))
        for candidate, score_dict in candidate_score_dict.items():
            bert_norm = score_dict["bert"]
            freq_norm = score_dict["freq"]
            len_norm = score_dict["len"]
            cos_norm = score_dict["cos"]
            lm_norm = score_dict["lm"]
            word_norm = score_dict["word"]
            sentence_norm = score_dict["sentence"]

            # Compute final score with normalized values
            final_score = (weight_bert * bert_norm +
                        weight_freq * freq_norm +
                        weight_len * len_norm +
                        weight_cos * cos_norm +
                        weight_lm * lm_norm + 
                        weight_word * word_norm + 
                        weight_sentence * sentence_norm)
            
            #if cos_norm == 0.0:
            #    final_score = 0.0

            score_list.append((candidate, final_score))

            index += 1
            print("Total candidate count: ", index, end="\r")

        # Store ranked suggestions
        normalized_ranked_dictionary[complex_word_key] = sorted(score_list, key=lambda x: x[1], reverse=True)
        best_suggestion = max(score_list, key=lambda x: x[1])[0]
        if max(score_list, key=lambda x: x[1])[1] == 0.0:
            print(f"No substitution with suitable pos tag found for the word {complex_word_key}, original word is being preserved...")
            best_suggestion = complex_word_key
        else:
            best_suggestion = max(score_list, key=lambda x: x[1])[0]
        #suggested_word[key] = best_suggestion
        suggested_token[complex_word_key] = best_suggestion

    #return suggested_word, normalized_ranked_dictionary, index
    return suggested_token, normalized_ranked_dictionary, index

def run_substitution_ranking(all_suggestion_scores,
                             df_subtlex, complex_words, eng_words, 
                             glove, 
                             weights):
    # Step 3: Apply normalized scores and finalize ranking
    #all_suggested_words = []
    all_suggested_tokens = []
    all_ranked_dictionaries = []

    print("Substitution ranking running...")
    index = 0
    for batch_idx, suggestion_score_dict in enumerate(all_suggestion_scores):
    #for batch_idx, (suggestion_dict, batch_scores) in enumerate(batch_results):
        #normalized_suggested_word, normalized_ranked_dictionary, index = rank_substitution_scores(
        normalized_suggested_token, normalized_ranked_dictionary, index = rank_substitution_scores(
            suggestion_score_dict, df_subtlex, complex_words, eng_words, glove, weights, index
        )
        #all_suggested_words.append(normalized_suggested_word)
        all_suggested_tokens.append(normalized_suggested_token)
        all_ranked_dictionaries.append(normalized_ranked_dictionary)

    print("\nSubstitution ranking complete.")
    #return all_suggested_words, all_ranked_dictionaries
    return all_suggested_tokens, all_ranked_dictionaries











def score_substitutions_old(all_suggestions, 
                        df_subtlex, complex_words, eng_words, 
                        all_cleaned_texts,
                        model, sentence_model, tokenizer, glove,
                        device):
    """
    Runs substitution_ranker on each batch individually and collects results.
    
    Args:
        all_suggestions (list of dict): List where each element is a dictionary of suggestions for a batch.
        all_cleaned_texts (list of str): List of cleaned texts, one per batch.

    Returns:
        all_suggested_words (list of dict): List of best substitutions per batch.
        all_ranked_dictionaries (list of dict): List of ranked suggestions per batch.
    """

    # Step 1: Collect all scores across all sentences before normalization
    global_score_dict = {
        "bert": [],
        "freq": [],
        "len": [],
        "cos": [],
        "lm": []
    }

    batch_results = []
    pos_flags = []
    word_count = 0

    print("Scores for suggestions being calculated...")

    for batch_idx, suggestion_dict in enumerate(all_suggestions):
        print(f"Suggestion {batch_idx+1}/{len(all_suggestions)}", end="\r")

        original_text = all_cleaned_texts[batch_idx]

        # Compute raw scores without normalization
        batch_scores, batch_pos_flags, word_count = collect_scores(
            suggestion_dictionary=suggestion_dict,
            df_subtlex=df_subtlex,
            complex_words=complex_words,
            eng_words=eng_words,
            original_text=original_text,
            model=model,
            sentence_model=sentence_model,
            tokenizer=tokenizer,
            glove=glove,
            device=device,
            word_count=word_count
        )

        # Collect scores for global normalization
        for key in global_score_dict.keys():
            global_score_dict[key].extend(batch_scores[key])
        pos_flags.extend(batch_pos_flags)

        # Store batch results for later normalization
        batch_results.append((suggestion_dict, batch_scores))

    print("\n")
    
    # Debug: Print statistics before normalization
    print("\n--- Raw Score Statistics Before Normalization ---")
    for key in global_score_dict.keys():
        print(f"{key} Score Statistics:")
        print(pd.DataFrame(global_score_dict[key]).describe())

    # Step 2: Normalize scores globally
    for key in global_score_dict.keys():
        global_score_dict[key] = np.array(global_score_dict[key])
        global_score_dict[key] = normalize_score(global_score_dict[key], score_type=key)

    # Debug: Print statistics after normalization
    print("\n--- Normalized Score Statistics ---")
    for key in global_score_dict.keys():
        print(f"{key} Normalized Score Statistics:")
        print(pd.DataFrame(global_score_dict[key]).describe())

    global_score_dict["pos_flag"] = pos_flags
    for key in global_score_dict.keys():
        print(f"Length of {key} scores: {len(global_score_dict[key])}")
    
    return global_score_dict




def collect_scores_old(suggestion_dictionary, df_subtlex, complex_words, eng_words, original_text, 
                                       model, sentence_model, tokenizer, glove, device, word_count):
    """
    Collects scores without normalization.
    """
    # Storage for raw scores before normalization
    score_list_dict = {"bert": [], "freq": [], "len": [], "cos": [], "lm": []}
    pos_flags = []

    # Compute scores for all suggestions
    for key, suggestions in suggestion_dictionary.items():
        candidate_parallel_list = []
        candidate_sentence_list = []
        for candidate, bert_score in suggestions:
            if candidate.startswith("##"):
                if not(key.startswith("##")):
                    continue
                else:
                    candidate_clean = candidate[2:]
                    key_clean = key[2:]
            else:
                if key.startswith("##"):
                    continue
                candidate_clean = candidate
                key_clean = key
            
            is_glove = isinstance(glove, torchtext.vocab.GloVe)
            use_freq = False
            is_complex =  ((is_glove and candidate_clean not in glove.stoi) or (use_freq and candidate_clean not in df_subtlex.iloc[:, 0].keys()) or (candidate_clean in complex_words))
            #is_complex =  ((is_glove and candidate_clean not in glove.stoi) or (use_freq and candidate_clean not in df_subtlex.iloc[:, 0].keys()) or (candidate_clean not in eng_words) or (candidate_clean in complex_words))
            if is_complex:
                continue
            word_count += 1
            print("Word count: ", word_count, end="\r")

            target_word_wo_occ = key.split("_[")[0] if "_[" in key else key
            occurrence_count = int(key.split("_[")[1][:-1]) if "_[" in key else 1
            target_word_clean = key_clean.split("_[")[0] if "_[" in key_clean else key_clean
            target_word = target_word_wo_occ.split()[0] if len(target_word_wo_occ.split()) > 1 else target_word_wo_occ

            candidate_parallel_list.append(candidate_clean)
            #candidate_sentence, _ = place_suggestion_into_sentence_token(original_text, target_word, candidate, tokenizer, occurrence_count)
            candidate_sentence, _ = place_suggestion_into_sentence_word(original_text, target_word, candidate, occurrence_count)
            candidate_sentence_list.append(candidate_sentence)

            # Compute individual scores
            lm_score = compute_lm_perplexity_loss(original_text, target_word, candidate, model, tokenizer, target_occurrence_count=occurrence_count, device=device) #uses original key for BERT tokenizer compatibility
            freq_score = compute_frequency_score(candidate_clean, df_subtlex)
            #freq_score = bert_word_similarity_score(target_word_clean, candidate_clean, model, tokenizer, device)
            len_score = compute_length_score(candidate_clean)
            #len_score = edit_distance_score(target_word_clean, candidate_clean) #levenstein score
            cos_score = compute_cosine_similarity(target_word_clean, candidate_clean, glove) #uses key_clean
            #cos_score = bert_sentence_similarity(target_word_clean, candidate_clean, sentence_model)
            #cos_score = word_movers_distance(target_word_clean, candidate_clean, word_vectors) #word movers distance score
            #context_score = compute_contextual_fit_score(original_text, target_word, candidate, model, tokenizer, target_occurence_count=occurrence_count)
            #cos_score = 

            # Check part-of-speech compatibility
            #pos_flag = check_pos(original_text=original_text, target_word=target_word_wo_occ, suggestion=candidate_clean, occurrence_count=occurrence_count)
            pos_flag = check_pos_bert(original_text=original_text, target_word=target_word_wo_occ, suggestion=candidate, tokenizer=tokenizer, occurrence_count=occurrence_count)
            #if check_pos(original_text=original_text, target_word=target_word_wo_occ, suggestion=candidate):
            #    pos_flag = True
            #else:
            #    pos_flag = False

            # Store raw scores
            score_list_dict["bert"].append(bert_score)
            score_list_dict["freq"].append(freq_score)
            score_list_dict["len"].append(len_score)
            score_list_dict["cos"].append(cos_score)
            score_list_dict["lm"].append(lm_score)
            pos_flags.append(pos_flag)
        #cos_score = bert_sentence_similarity([original_text], candidate_parallel_list, sentence_model)
        #sentence_similarity = bert_sentence_similarity([original_text], candidate_sentence_list, sentence_model)
        #score_list_dict["freq"].extend(sentence_similarity)
        #word_similarity = bert_sentence_similarity([target_word_clean], candidate_parallel_list, sentence_model)
        #score_list_dict["freq"].extend(word_similarity)
    return score_list_dict, pos_flags, word_count



def rank_substitution_scores_old(suggestion_dict, df_subtlex, complex_words, eng_words, glove, global_score_dict, weights, index):
    """
    Finalizes the scores and ranks the suggestions based on normalized scores.
    """
    weight_bert, weight_cos, weight_lm, weight_freq, weight_len = weights

    #suggested_word = {}
    suggested_token = {}
    normalized_ranked_dictionary = {}

    for key, suggestions in suggestion_dict.items():
        score_list = []
        for candidate, _ in suggestions:
            if candidate.startswith("##"):
                if not(key.startswith("##")):
                    continue
                else:
                    candidate_clean = candidate[2:]
                    key_clean = key[2:]
            else:
                if key.startswith("##"):
                    continue
                candidate_clean = candidate
                key_clean = key
            
            is_glove = isinstance(glove, torchtext.vocab.GloVe)
            use_freq = False
            is_complex = ((is_glove and candidate_clean not in glove.stoi) or (use_freq and candidate_clean not in df_subtlex.iloc[:, 0].keys()) or (candidate_clean in complex_words))
            #is_complex =  ((is_glove and candidate_clean not in glove.stoi) or (use_freq and candidate_clean not in df_subtlex.iloc[:, 0].keys()) or (candidate_clean not in eng_words) or (candidate_clean in complex_words))
            if is_complex:
                continue
            #if candidate not in glove.stoi or candidate not in df_subtlex.iloc[:, 0].keys() or candidate in complex_words:
            #    continue
            print(f"Normalizing index {index}/{len(global_score_dict['bert'])-1}...", end="\r")

            bert_norm = global_score_dict["bert"][index]
            freq_norm = global_score_dict["freq"][index]
            len_norm = global_score_dict["len"][index]
            cos_norm = global_score_dict["cos"][index]
            lm_norm = global_score_dict["lm"][index]
            pos_flag = global_score_dict["pos_flag"][index]

            if pos_flag:
                # Compute final score with normalized values
                final_score = (weight_bert * bert_norm +
                            weight_freq * freq_norm +
                            weight_len * len_norm +
                            weight_cos * cos_norm +
                            weight_lm * lm_norm)
            else:
                final_score = 0.0

            score_list.append((candidate, final_score))

            index += 1

        # Store ranked suggestions
        normalized_ranked_dictionary[key] = sorted(score_list, key=lambda x: x[1], reverse=True)
        if score_list:
            best_suggestion = max(score_list, key=lambda x: x[1])[0]
            if max(score_list, key=lambda x: x[1])[1] == 0.0:
                print(f"No substitution with suitable pos tag found for the word {key}, original word is being preserved...")
                best_suggestion = key
            else:
                best_suggestion = max(score_list, key=lambda x: x[1])[0]
            #suggested_word[key] = best_suggestion
            suggested_token[key] = best_suggestion

    #return suggested_word, normalized_ranked_dictionary, index
    return suggested_token, normalized_ranked_dictionary, index


def run_substitution_ranking_old(all_suggestions, global_score_dict,
                             df_subtlex, complex_words, eng_words, 
                             glove, 
                             weights):
    # Step 3: Apply normalized scores and finalize ranking
    #all_suggested_words = []
    all_suggested_tokens = []
    all_ranked_dictionaries = []

    print("Substitution ranking running...")
    index = 0
    for batch_idx, suggestion_dict in enumerate(all_suggestions):
    #for batch_idx, (suggestion_dict, batch_scores) in enumerate(batch_results):
        #normalized_suggested_word, normalized_ranked_dictionary, index = rank_substitution_scores(
        normalized_suggested_token, normalized_ranked_dictionary, index = rank_substitution_scores(
            suggestion_dict, df_subtlex, complex_words, eng_words, glove, global_score_dict, weights, index
        )
        #all_suggested_words.append(normalized_suggested_word)
        all_suggested_tokens.append(normalized_suggested_token)
        all_ranked_dictionaries.append(normalized_ranked_dictionary)

    print("\nSubstitution ranking complete.")
    #return all_suggested_words, all_ranked_dictionaries
    return all_suggested_tokens, all_ranked_dictionaries
