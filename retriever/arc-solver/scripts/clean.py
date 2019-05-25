import json
import os

suffix = ".jsonl"
clean_suffix = ".clean.jsonl"

for mode in ["Train", "Dev", "Test"]:

    prefix = "data/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-{}-question-reform.nn".format(mode)
    in_file = prefix + suffix
    out_file = prefix + clean_suffix
 
    f1 = open(in_file, 'r')
    f2 = open(out_file, 'w')

    for l in f1.readlines():
        d = json.loads(l.strip())
        choices = d["question"]["choices"]
        if len(choices) == 4:
            f2.write(l)
        elif len(choices) == 3:
            choices.append({"text":"...","label":"D"})      
            d["question"]["choices"] = choices
            f2.write(json.dumps(d) + '\n')
 
    f1.close()
    f2.close()

