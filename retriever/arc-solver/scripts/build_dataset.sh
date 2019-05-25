#!/bin/bash
set -e

# ----------------------------------------
# Build training set for BiDAF model on the QA dataset
# ----------------------------------------

input_file=$1
query_mode=$2
# Set this to name your run
run_name=default

if [ -z $input_file ] ; then
  echo "USAGE: ./scripts/build_train_bidaf.sh question_file.jsonl"
  exit 1
fi

input_file_prefix=${input_file%.jsonl}

# File containing retrieved hits per choice (using the key "support")
input_file_with_hits=${input_file_prefix}_with_hits_${run_name}.jsonl

echo $input_file
echo $input_file_with_hits

# File with all the hits combined per question (using the key "para")
bidaf_input=${input_file_prefix}_with_paras_${run_name}.jsonl

# Collect hits from ElasticSearch for each question + answer choice
if [ ! -f ${input_file_with_hits} ]; then
	python arc_solvers/processing/add_retrieved_text.py \
		${input_file} \
		${input_file_with_hits} \
                ${query_mode}
fi

# Merge hits for each question
if [ ! -f ${bidaf_input} ]; then
	python arc_solvers/processing/convert_to_para_comprehension.py \
	${input_file_with_hits} \
	${input_file} \
	${bidaf_input}
fi

