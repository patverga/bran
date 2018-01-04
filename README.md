# CDR_IE

Requirements
----------
tensorflow version 1.0.1


Environment Variables
------
export CDR_IE_ROOT=`pwd`   


Process CDR Data
------
Process the CDR dataset   
`${CDR_IE_ROOT}/bin/process_CDR.sh` 

Process the CDR dataset including additional weakly labeled data   
`${CDR_IE_ROOT}/bin/process_CDR_extra_data.sh`  
 

Generate Full CTD dataset
-----
`${CDR_IE_ROOT}/bin/generate_full_CTD_data.sh`


Run Model
------
Train a model locally on gpu id 0  
`${CDR_IE_ROOT}/bin/run.sh ${CDR_IE_ROOT}/configs/cdr/relex/cdr_2500 0`   

Train a model on gypsum   
`${CDR_IE_ROOT}/bin/srun.sh ${CDR_IE_ROOT}/configs/cdr/relex/cdr_2500`   


Saving loading models
-----
By default the model will be evaulated on the CDR dev set. To save the best model to the file 'model.tf', add the save_model flag   
`${CDR_IE_ROOT}/bin/run.sh ${CDR_IE_ROOT}/configs/cdr/relex/cdr_2500 0 --save_model model.tf`   
 
To load a saved model, run   
`${CDR_IE_ROOT}/bin/run.sh ${CDR_IE_ROOT}/configs/cdr/relex/cdr_2500 0 --load_model path/to/model.tf `   
 
 
