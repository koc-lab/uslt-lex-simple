# uslt-lex-simple

This repository contains the new implementation of "Unsupervised Simplification of Legal Texts" (USLT) submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing. 

For earlier versions, you can refer to [https://github.com/koc-lab/lex-simple](https://github.com/koc-lab/lex-simple). 

## lex-simple dataset

As a remedy to the lack of a parallel complex-to-simple text simplification corpus in the legal domain, we have constructed the lex-simple dataset, containing 500 legal sentences from the US Supreme Court Cases corpora. By collaborating with the faculty and the students of Bilkent Law School, we produced 3 different simplified reference files for these 500 sentences. We hope that this dataset can serve as a benchmark for future legal text simplification studies. `lex-simple-dataset` folder contains the `full` version of the dataset, as well as the `train`, `test` and `val` splits we use in our work.

## Installation

1. Cloning the repository
   ```bash
   git clone https://github.com/koclab/uslt-lex-simple.git

2. Installing the required dependencies

   We recommend to create a Conda environment or a virtual environment and install the dependencies to the environment. 
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, the dependencies can also be installed from the .yml file:
   ```bash
   conda env create -f environment.yml
   ```

## Data Loading

Loading of word statistics that are used in CWI, loading of the original legal sentences from the .txt files, loading of the CWI data, and a custom `MaskedSentenceDataset` class to handle CWI data in batches are contained in the `load_data.py` file.

## Running USLT

To run USLT with all of the complex word identification (CWI), suggestion generation (SG) and substitution ranking (SR) steps, you can directly use the main script:

   ```bash
   python main.py --gpu_device="cuda:0" --data_split="test"
   ```

`main.py` will produce the output simplification texts for the USLT.

Notice that there are commented scripts in the `main.py` file. You can use one of these based on the purpose you are using this file, and comment the rest of the code snippets. These serve the following purposes:

### Main Option 1: Ablation Study

If you uncomment this option, you will be able to conduct ablation study by dropping one feature at a time for each of the features contributing to the weighted summation in the SR step. The outputs will be written inside the `ablation_output_data` folder.

### Main Option 2: Manually Entered Weights

If you use this option, you are expected to manually enter the weights that you want to use in the SR step. Currently, the optimal weights found through the hyperparameter optimization are entered inside the code file.

### Main Option 3: Using Logs From Hyperparameter Optimization

This snippet can be used only if you have some logs of the hyperparameter optimization process. This snippet will check some of the logs entered during hyperparameter optimization and run USLT using these weights. Then, these trials will be logged inside the folder `output_data`. Note that you might need to create additional folder inside `output_data/{data_split}/` as `trial_{opt_src}`, where `{opt_src}` is the metric for which you conduct the hyperparameter optimization through.

## USLT Components

USLT is mainly composed of the CWI, SR, and SG components.

### Complex Word Identification (CWI)
The code or the CWI stage can be found in the `construct_masked_lm.py` script. 

### Suggestion Generation (SG)
The code for the SG step can be found in the `suggestion_generator.py` file.

### Substitution Ranking (SR)
The code for the SR step is contained in two files. First, the `ranking_scores.py` is used to compute the ranking scores for each feature. Then, the candidate ranking is performed using the functions in the `substitution_ranker.py` file. 

### Constructing the Simplification Output

The functions in the `simplify.py` file are used to construct the final text simplification output and write them into files in the `output_data` folder.

## Hyperparameter Optimization

The logs of our Bayesian optimization procedure can be found in the `optimization_log_caselaw_onlybert.csv` file. To run the hyperparameter optimization script, you can use the following line of command:

   ```bash
   python hyperparameter_opt.py --dataset="caselaw" --use_training_data="False" --gpu_device="cuda:0"
   ```

The arguments can be changed to customize the hyperparameter optimization process. Note that currently, the hyperparameter optimization is conducted on the BERTScore metric. If wanted to conduct using another metric (e.g. DC score), make sure to change the `LOG_FILE` variable at the top of the `bayesian_opt.py` file. Make sure that the optimization objective is also defined accordingly.

## Sentence Splitting
Sentence splitting process is independent from the operations in the CWI, SG and SR steps, and is not implemented as part of the `main.py` script. To do structural simplification on top of the lexical simplification steps through sentence splitting, you can follow the steps in [https://github.com/Lambda-3/DiscourseSimplification/tree/master](https://github.com/Lambda-3/DiscourseSimplification/tree/master). The repo outlines the setup of Maven and related dependencies for the sentence splitting repo to function. We have also forked the original repository in [https://github.com/koc-lab/SentenceSplitting.git] (https://github.com/koc-lab/SentenceSplitting.git). Our explanations will differ only from that of the original repo in terms of the repo name, where we will use `SentenceSplitting` instead of `DiscourseSimplification`. 

In particular, run the following lines of code:

   ```bash
   git clone https://github.com/koc-lab/SentenceSplitting.git
   cd SentenceSplitting
   mvn clean install -DskipTests
   ```

Create the directory `edu/stanford/nlp/models/pos-tagger/english-left3words/` inside the `SentenceSplitting` folder. Move the stanford nlp taggers `english-caseless-left3words-distsim.tagger` and `english-left3words-distsim.tagger` inside the folder you have created. You can find in this drive link inside these folders: [https://drive.google.com/drive/folders/1GQerFiPgzFnS2lawIfAz8C_NsLbdQUJG?usp=share_link](https://drive.google.com/drive/folders/1GQerFiPgzFnS2lawIfAz8C_NsLbdQUJG?usp=share_link). Then, generate an empty file called 'input.txt' inside this directory and copy and paste the lexically simplified document generated by the USLT scripts.

Finally, inside the `SentenceSplitting` folder, run:

   ```bash
   mvn clean compile exec:java
   cd ..
   python decode_sentence_splitting.py
   ```

You have generated the output of USLT-ss, the USLT variant that performs sentence splitting on top of lexical simplification. The file is saved inside the `output_data` folder.

## Evaluation

Either after running USLT, or using the uploaded simplification outputs, you can run the evaluation script to assess the performance of each simplification model:

   ```bash
   python eval.py
   ```

## Replicating Ablation Studies

To replicate the results of the ablation study, you can run the `eval_ablation.py` script:

   ```bash
   python eval_ablation.py
   ```

## Human Evaluation

On top of the evaluations with the automated metrics, we have also perfomed a human evaluation study by consulting with a legal expert to assess whether the outputs of USLT and other text simplification methods aligned with human judgment. The results of our study can be found in the `results_human_eval.csv` file.

For further details on the application we used in the human evaluation study and to use it in your own research, you can consult to the repo in [this link](https://github.com/EralpK/uslt-human-eval-app).
