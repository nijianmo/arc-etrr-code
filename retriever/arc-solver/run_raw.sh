DATA_PATH="./data"

# challenge set
sh scripts/build_dataset.sh \
    $DATA_PATH/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Train.jsonl common 

sh scripts/build_dataset.sh \
    $DATA_PATH/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Dev.jsonl common

sh scripts/build_dataset.sh \
    $DATA_PATH/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Test.jsonl common 

# easy set
#sh scripts/build_dataset.sh \
#    $DATA_PATH/ARC-V1-Feb2018/ARC-Easy/ARC-Easy-Train.jsonl 

#sh scripts/build_dataset.sh \
#    $DATA_PATH/ARC-V1-Feb2018/ARC-Easy/ARC-Easy-Dev.jsonl 

#sh scripts/build_dataset.sh \
#    $DATA_PATH/ARC-V1-Feb2018/ARC-Easy/ARC-Easy-Test.jsonl 

