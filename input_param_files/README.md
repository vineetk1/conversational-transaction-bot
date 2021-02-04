# Parameters to train, validate, and test models 
## Location of log and checkpoint files
### &emsp; &emsp; Path to ctb logs files 
*ctb_logs* file is in the the default (i.e.  conversational-transaction-bot) directory.
### &emsp; &emsp; Path to TensorBoard logs files
It includes the following directories:
1. tensorboard_logs directory in the default (i.e.  conversational-transaction-bot) directory.
1. Model-type and tokenizer-type.
1. A unique version number that increases every time training is done.   

Following is an example: tensorboard_logs/model_type=distilgpt2-dstc2,tokenizer_type=gpt2-dstc2/version_0
### &emsp; &emsp; Path to Checkpoint files
It includes the following directories:
1. Path of TensorBoard logs files.
1. checkpoints.   

Following is an example: tensorboard_logs/model_type=distilgpt2-dstc2,tokenizer_type=gpt2-dstc2/version_0/checkpoints   
### &emsp; &emsp; Name of Checkpoint files
When training stops, the state of the model is checkpointed in the file *last.ckpt*. Additionally, epochs with the lowest validation loss are also checkpointed. The names of files with lowest validation loss includes the following:
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
* test_pass_fail_stat (bool, optional) --  collect statistics on the trained model when no_testing=False (Default: False)
### &emsp; &emsp; Parameters used in python-dictionary 1   
* model_type (str, optional if "chkpt" is specified in python_dictionary #0) -- name of model and dataset to load   
* tokenizer_type (str, optional if "chkpt" is specified in python_dictionary #0) -- name of tokenizer and dataset to load   
* optz (see PyTorch documentation, optional if training is resumed) -- name of optimizer   
* optz_params (see PyTorch documentation, optional) -- hyper-parameters of optimizer EXCLUDING "params"   
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
## Scenarios:
### &emsp; &emsp; Fine-tune a pre-trained model
Note that model_type=distilgpt2-dstc2 means that the DistilGPT2 pretrained model will be loaded and it will be fine-tuned with the DSTC2 dataset   
Note that tokenizer_type=gpt2-dstc2 means that the GPT2 tokenizer will be loaded and it will be fine-tuned with tokens from the DSTC2 dataset   
{'save_top_k': 2, 'no_testing': True}   
{'model_type': 'distilgpt2-dstc2', 'tokenizer_type': 'gpt2-dstc2', 'optz': 'Adam', 'optz_params': {'lr': 1e-06}, 'lr_sched': 'ReduceLROnPlateau', 'lr_sched_params': {'mode': 'min', 'patience': 11, 'factor': 0.001}}   
{'default_format_path': 'data/dialog-bAbI-tasks/dstc2/defaultFormat.train'}   
{'gpus': 1, 'auto_scale_batch_size': True}   
### &emsp; &emsp; Resume training the model from where it was stopped
{'save_top_k': 3, 'test_pass_fail_stat': True}     
{'model_type': 'distilgpt2-dstc2', 'tokenizer_type': 'gpt2-dstc2'}       
{'default_format_path': 'data/dialog-bAbI-tasks/dstc2/defaultFormat.train'}   
{'gpus': 1, 'resume_from_checkpoint': 'tensorboard_logs/model_type=distilgpt2-dstc2,tokenizer_type=gpt2-dstc2/version_17/checkpoints/last.ckpt'}   
### &emsp; &emsp; Continue training with different hyper-parameters the checkpoint model that has the lowest validation loss
{'save_top_k': 4, 'no_testing': True, 'chkpt': 'tensorboard_logs/model_type=distilgpt2-dstc2,tokenizer_type=gpt2-dstc2/version_20/checkpoints/optz=Adam,lr=1e-06,lr_sched=ReduceLROnPlateau,mode=min,patience=11,factor=0.001,verbose=True,epoch=22-val_loss=0.15321.ckpt'}    
{'optz': 'SGD', 'optz_params': {'lr': 0, 'momentum': 0.9, 'nesterov': True}, 'lr_sched': 'CyclicLR', 'lr_sched_params': {'base_lr': 1e-12, 'max_lr': 1e-2}}   
{'default_format_path': 'data/dialog-bAbI-tasks/dstc2/defaultFormat.train', 'batch_size': 2}   
{'gpus': 1}   
### &emsp; &emsp; Test a pre-trained model with DSTC2 dataset
{'no_training': True, 'test_pass_fail_stat': True}    
{'model_type': 'distilgpt2-dstc2', 'tokenizer_type': 'gpt2-dstc2'}    
{'default_format_path': 'data/dialog-bAbI-tasks/dstc2/defaultFormat.train', 'batch_size': 2}   
{'gpus': 1}   
### &emsp; &emsp; Test a checkpoint model with DSTC2 dataset
{'no_training': True, 'chkpt': 'tensorboard_logs/model_type=distilgpt2-dstc2,tokenizer_type=gpt2-dstc2/version_20/checkpoints/optz=Adam,lr=1e-06,lr_sched=ReduceLROnPlateau,mode=min,patience=11,factor=0.001,verbose=True,epoch=22-val_loss=0.15321.ckpt'}   
{}   
{'default_format_path': 'data/dialog-bAbI-tasks/dstc2/defaultFormat.train', 'batch_size': 2}   
{'gpus': 1}   
