DATA_PATH="./data/ARC-V1-Feb2018"
query_mode="reform"

# clean the data--- removing questions with more than 4 choices
python ./scripts/clean.py

# retrieve paragraphs and construct new qa files
input_file=$DATA_PATH"/ARC-Challenge/ARC-Challenge-Train-question-reform.nn.clean.jsonl"
input_file_with_hits=$DATA_PATH"/ARC-Challenge/ARC-Challenge-Train-question-reform.nn-qa-para.clean.jsonl"

# Collect hits from ElasticSearch for each question + answer choice
if [ ! -f ${input_file_with_hits} ]; then
        python arc_solvers/processing/convert_to_qa_para.py \
                ${input_file} \
                ${input_file_with_hits} \
                ${query_mode}
fi

input_file=$DATA_PATH"/ARC-Challenge/ARC-Challenge-Dev-question-reform.nn.clean.jsonl"
input_file_with_hits=$DATA_PATH"/ARC-Challenge/ARC-Challenge-Dev-question-reform.nn-qa-para.clean.jsonl"

# Collect hits from ElasticSearch for each question + answer choice
if [ ! -f ${input_file_with_hits} ]; then
        python arc_solvers/processing/convert_to_qa_para.py \
                ${input_file} \
                ${input_file_with_hits} \
                ${query_mode}
fi

input_file=$DATA_PATH"/ARC-Challenge/ARC-Challenge-Test-question-reform.nn.clean.jsonl"
input_file_with_hits=$DATA_PATH"/ARC-Challenge/ARC-Challenge-Test-question-reform.nn-qa-para.clean.jsonl"

# Collect hits from ElasticSearch for each question + answer choice
if [ ! -f ${input_file_with_hits} ]; then
        python arc_solvers/processing/convert_to_qa_para.py \
                ${input_file} \
                ${input_file_with_hits} \
                ${query_mode}
fi

