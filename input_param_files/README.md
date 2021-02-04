Note the following:
	(1) This file name should be last in the command-line.
	(2) Do NOT change the order of python-dictionaries in this file.
	(3) The default directory is "conversational-transaction-bot"
 
## Command-line:
python3 ctbMain.py input_param_files/distilgpt2_dstc2 
## Path to ctb logs files:
It is the default directory, and the name of the file is "ctb_logs".
## Path to TensorBoard logs files:
It includes the following directories:
1. tensorboard_logs.
1. Model-type and tokenizer-type.
1. A unique version number that increases every time training is done.   
Following is an example: tensorboard_logs/model_type=distilgpt2-dstc2,tokenizer_type=gpt2-dstc2/version_0
## Path to Checkpointed files:
It includes the following directories:
1. Path of TensorBoard logs files.
1. Checkpoints.   

Following is an example: tensorboard_logs/model_type=distilgpt2-dstc2,tokenizer_type=gpt2-dstc2/version_0/checkpoints
## Name of Checkpointed files:
During training, the last epoch is always checkpointed in the file *last.ckpt*. 
Additionally, epochs with the lowest validation loss are also checkpointed. The
names of these files includes the following:
1. Optimizer and its parameters.
1. LR-scheduler and its parameters.
1. Epoch number plus the validation loss.    

Following is an example: optz=SGD,lr=1.67017e-05,momentum=0.9,nesterov=True,lr_sched=ReduceLROnPlateau,mode=min,patience=9,epoch=00-val_loss=1.15033.ckpt
## Parameters
### &emsp; &emsp; Parameters used in python-dictionary 0   
* save_top_k (int, optional) -- number of checkpoint files to save (Default: 1)   
* chkpt (str, optional) -- path to checkpoint file that will be loaded   
* no_training (bool, optional) -- do not train the model (Default: False) 
* no_testing (bool, optional) -- do not test the model (Default: False)   
* test_pass_fail_stat (bool, optional) --  whether to collect statistics on the trained model \[no_testing=False] (Default: False)
### &emsp; &emsp; Parameters used in python-dictionary 1   
* model_type (str) -- name of model and dataset to load   
* tokenizer_type (str) -- name of tokenizer and dataset to load   
* optz (see PyTorch documentation, optional if training is resumed) -- name of optimizer   
* optz_params (see PyTorch documentation, optional if training is resumed) -- hyper-parameters of optimizer EXCLUDING "params"   
* lr_sched (see PyTorch documentation, optional) -- hyper-parameters of lr-scheduler   
* lr_sched_params (see PyTorch documentation, optional) -- hyper-parameters of scheduler EXCLUDING "optimizer"      
### &emsp; &emsp; Parameters used in python-dictionary 2   
* default_format_path (str) -- path to training dataset  
* batch_size (int, optional) -- batch size (Default: 2)  
### &emsp; &emsp; Parameters used in python-dictionary 3
See Lightning Trainer documentation for parameters that can be used in this python-dictionary. Some parameters are listed as follows:   
* gpus (Union\[int, str, List\[int], None]) –- number of gpus to train on (int) or which GPUs to train on (list or str) applied per node   
* auto_lr_find (bool, optional) -- automatically find the initial learning-rate (Default: False)   
* auto_scale_batch_size (bool, optional) -- automatically find the batch-size (Default: False)   
## Scenario:
Stop training and resume training using same hyperparams. Stop with ctrl-c
Stop training and load_checkpoint and change hyperparams.
