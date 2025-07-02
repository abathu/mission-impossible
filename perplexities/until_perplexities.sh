#!/bin/sh

conda activate mistral

echo "Running perplexities for seed 41 with randinit and no pos encodings"
echo "Command: python /home/s2678328/mission-impossible_fork/perplexities/perplexities.py shuffle_nondeterministic shuffle_nondeterministic 100M 41 randinit -np"

# python /home/s2678328/mission-impossible_fork/perplexities/perplexities.py shuffle_control shuffle_control 100M 41 randinit -np 

python /home/s2678328/mission-impossible_fork/perplexities/perplexities.py \
    shuffle_control shuffle_control 100M 41 randinit  /home/s2678328/BabyLM_dataset/translated_dev_file/Russain 

echo "DONE"


# /home/s2678328/nlp/llms-in-llms/babylm_models/babylm_shuffle_nondeterministic_100M_randinit