Vineet Kumar, sioom.ai

This input file has user-settable hyper-parameters for training and testing
  a model.

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

## Scenario:
Stop training and resume training using same hyperparams. Stop with ctrl-c
Stop training and load_checkpoint and change hyperparams.

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
* default_format_path': 'path to training dataset'
- 
{'default_format_path': 'data/dialog-bAbI-tasks/dstc2/defaultFormat.train', 'batch_size': 2} 


parameters for Lightning Trainer 
- For a list of parameters, see Trainer.__init__(...) in PyTorch Lightning documentation
- The following will make Lightning fail because of a bug: 
     'auto_lr_find': True, 'auto_scale_batch_size': True
- 
{'gpus': 1, 'auto_lr_find': False, 'auto_scale_batch_size': False}

