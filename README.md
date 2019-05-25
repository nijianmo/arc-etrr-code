### arc-etrr-code
This is the code for our NAACL 19' work
- Learning to Attend On Essential Terms: An Enhanced Retriever-Reader Model for Open-domain Question Answering, Jianmo Ni, Chenguang Zhu, Weizhu Chen and Julian McAuley, NAACL 2019.

This repo follows the following hierarchy:
```
arc-etrr-code
|---retriever
  |---selector
  |---arc-solver
|---reader
|---data

```

### selector
This is the proposed essential term selector. Please ```cd selector``` into that folder.

1. Train the model on the labeled dataset.
We provide the preprocessed dataset in the data folder--- check out files ending with ```-processed.json```. If you choose to preprocess dataset by yourself, please run ./download.sh to download Glove embeddings and ConceptNet, and then run ./run.sh to preprocess dataset and train the model.

Specifically, the script will (1) preprocess the conceptnet, (2) preprocess the labeled dataset, and (3) start training the model. 

2. Apply the trained model on ARC dataset to identify essential terms, and update the raw ARC json data with a new (key,value) pair--- (```question_reform```, the essential terms).
We also provide the new ARC json data with the essential terms. If you choose to preprocess the dataset and run inference by your self, please replace the trained model name and run ./inference.sh.

### arc-solver
This is the arc-solver provided by ai2. Basically it starts elastic search and provides query search interface. Please ```cd arc-solver``` into the folder.
1. Download the data and models into data/ folder. This will also build the ElasticSearch index (assumes ElasticSearch 6+ is running on ES_HOST machine defined in the script)
```sh scripts/download_data.sh```

2. Retrieve evidence for each questions and build the new input qa files. 
```
sh run_reform.sh
```

This command will produce the files ```ARC-Challenge-*-question-reform.nn-qa-para.clean.jsonl```, which are then fed into the reader.

### reader
This is the proposed reader to predict the answer for the multiple-choice questions based on the question, choices and paragraphs. Please ```cd reader``` into that folder.
The reader and the selector are built upon the same code base from [Yuanfudao](https://github.com/intfloat/commonsense-rc). Hence the training and inference are similar to the sereader
1. Train the model on the ARC dataset.
Similarly, we provide the preprocessed dataset in the [data](https://drive.google.com/drive/folders/1u-7HwjjehLjQ2b-qe8qhLjTRflcJfAO2?usp=sharing) folder. Download the folder and put it under the reader folder. 
If you choose to preprocess dataset by yourself, please run ./run.sh to preprocess dataset. After obtaining the data, you can start to train the model. 
The trained model will be stored in the checkpoint folder. We also provide the pre-trained model [here](https://drive.google.com/open?id=19IKe3tZsRLl2wwdpxxWTvNPY20oScQ8S).

### Requirements
- PyTorch=0.4

Please cite our paper if you find the data and code helpful, thanks!
```
@inproceedings{Ni2018LearningTA,
  title={Learning to Attend On Essential Terms: An Enhanced Retriever-Reader Model for Open-domain Question Answering},
  author={Jianmo Ni and Chenguang Zhu and Weizhu Chen and Julian McAuley},
  booktitle={NAACL},
  year={2019}
}
```

