# Full Abstract Relation Extraction from Biological Texts with Bi-affine Relation Attention Networks

This code was used in the paper:  

Simultaneously Self-attending to All Mentions for Full-Abstract Biological Relation Extraction  
Patrick Verga , Emma Strubell, and Andrew McCallum.  
North American Chapter of the Association for Computational Linguistics (NAACL) 2018


# Requirements  
python version 2.7
tensorflow version 1.0.1


## Setup Environment Variables  
From this directory call: 
`source set_environment.sh`  
Note: this will only set the paths for this session. 


# Processing Data  
## CDR  
Process the CDR dataset   
`${CDR_IE_ROOT}/bin/process_CDR/process_CDR.sh` 

Process the CDR dataset including additional weakly labeled data   
`${CDR_IE_ROOT}/bin/process_CDR/process_CDR_extra_data.sh`  

These scripts will use byte-pair encoding (BPE) tokenization. There are also scripts to tokenize using the Genia tokenizer.

# Run Model  
Train a model locally on gpu id 0  
`${CDR_IE_ROOT}/bin/run.sh ${CDR_IE_ROOT}/configs/cdr/relex/cdr_2500 0`   

If you are using a cluster with Slurm, you can instead use this command:   
`${CDR_IE_ROOT}/bin/srun.sh ${CDR_IE_ROOT}/configs/cdr/relex/cdr_2500`   


## Saving loading models  
By default the model will be evaulated on the CDR dev set. To save the best model to the file 'model.tf', add the save_model flag   
`${CDR_IE_ROOT}/bin/run.sh ${CDR_IE_ROOT}/configs/cdr/relex/cdr_2500 0 --save_model model.tf`   
 
To load a saved model, run   
`${CDR_IE_ROOT}/bin/run.sh ${CDR_IE_ROOT}/configs/cdr/relex/cdr_2500 0 --load_model path/to/model.tf `  
 
## Pretrained Models
You can download some pretrained models [here](https://goo.gl/X9umaB)
 
 
## Generating the CTD dataset  
This script will generate the full CTD dataset. The following command will tokenize using BPE with a budget of 50k tokens.  
`${CDR_IE_ROOT}/bin/process_CTD/generate_full_CTD_data.sh`  

You can also generate the data using the genia tokenizer with   
`${CDR_IE_ROOT}/bin/process_CTD/generate_full_CTD_data_genia.sh`  

By default, abstracts with > 500 tokens are discarded. To not filter you can change the MAX_LEN variable to a very large number. 

