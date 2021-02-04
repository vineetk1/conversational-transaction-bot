Vineet Kumar, sioom.ai

This input file has user-settable hyper-parameters for training and testing
  a model.

Note the following:
	(1) This file name should be last in the command-line.
	(2) Do NOT change the order of python-dictionaries in this file.
	(3) The default directory is "conversational-transaction-bot"
 
Command-line:
-------------
python3 ctbMain.py input_param_files/distilgpt2_dstc2 

Path to ctb logs files:
-----------------------
It is the default directory, and the name of the file is "ctb_logs".

Path to TensorBoard logs files:
-----------------------------
It includes the following directories: (i) tensorboard_logs, (ii) model-type
and tokenizer-type, (iii) a unique version number that increases every time
training is done. Following is an example:
tensorboard_logs/model_type=distilgpt2-dstc2,tokenizer_type=gpt2-dstc2/version_0

Path to Checkpointed files:
---------------------------
It includes the following directories: (i) path of TensorBoard logs files,
(ii) checkpoints. Following is an example:
tensorboard_logs/model_type=distilgpt2-dstc2,tokenizer_type=gpt2-dstc2/version_0/checkpoints

Name of Checkpointed files:
---------------------------
During training, the last epoch is always checkpointed in the file 'last.ckpt'. 
Additionally, epochs with the lowest validation loss are also checkpointed. The
names of these files includes the following: (i) optimizer and its parameters,
(ii) lr-scheduler and its parameters, (iii) epoch number plus the validation loss.
Following is an example:
optz=SGD,lr=1.67017e-05,momentum=0.9,nesterov=True,lr_sched=ReduceLROnPlateau,mode=min,patience=9,epoch=00-val_loss=1.15033.ckpt

Scenario:
----------
Stop training and resume training using same hyperparams. Stop with ctrl-c
Stop training and load_checkpoint and change hyperparams.


parameters for file "ctbMain.py"
- 'save_top_k':    	 number of checkpointed files to save		 Default: 1
- 'chkpt': 		 'path to checkpoint file that will be loaded'
			    Path is not provided if a Huggingface pre-trained model is     
			    loaded or a custom-made model is initialized
- 'no_training':   	 whether to not train a model			 Default: False
- 'no_testing':    	 whether to not test a trained model		 Default: False
- 'test_pass_fail_stat': whether to collect statistic on a trained model Default: False
                         this parameter is active only when 'no_testing': False
-
{'save_top_k': 2, 'no_testing': True}


parameters for file "ctbModel.py"
- 'model_type':      'name of pretrained model plus name of training dataset'
- 'tokenizer_type':  'name of pretrained tokenizer plus name of training
                         dataset whose tokens are added'
- 'optz':            'name of optimizer'  {optional if resume_training}
- 'optz_params':     {parameters of optimizer EXCLUDING "params"} {optional}
- 'lr_sched':        'name of lr-scheduler' {optional}
- 'lr_sched_params': {parameters of scheduler EXCLUDING "optimizer"} {optional}
- eu = euler's number; e = exponent with a base of 10
  eu1 =   2.71828182846     eu-1 =  3.6787944117e-1     eu-2 =  1.3533528323e-1
  eu-3 =  4.978706836e-2    eu-4 =  1.831563888e-2      eu-5 =  6.73794699e-3
  eu-6 =  2.47875217e-3     eu-7 =  9.1188196e-4        eu-8 =  3.3546262e-4
  eu-9 =  1.234098e-4       eu-10 = 4.539992e-5         eu-11 = 1.67017e-5 
  eu-12 = 6.14421e-6        eu-13 = 2.2603294e-6        eu-14 = 8.31528719e-7
- 
{'model_type': 'distilgpt2-dstc2', 'tokenizer_type': 'gpt2-dstc2', 'optz': 'Adam', 'optz_params': {'lr': 8.31528719e-7}} 
#{'model_type': 'distilgpt2-dstc2', 'tokenizer_type': 'gpt2-dstc2', 'optz': 'Adam', 'optz_params': {'lr': 9.120108393559096e-08}, 'lr_sched': 'ReduceLROnPlateau', 'lr_sched_params': {'mode': 'min', 'patience': 11, 'factor': 0.001, 'verbose': True}} 
#{'model_type': 'distilgpt2-dstc2', 'tokenizer_type': 'gpt2-dstc2', 'optz': 'SGD', 'optz_params': {'lr': 1.831563888e-2, 'momentum': 0.9, 'nesterov': True}, 'lr_sched': 'ReduceLROnPlateau', 'lr_sched_params': {'mode': 'min', 'patience': 9}}
# If lr_sched=CyclicLR, then base_lr is the initial learning-rate
#{'model_type': 'distilgpt2-dstc2', 'tokenizer_type': 'gpt2-dstc2', 'optz': 'SGD', 'optz_params': {'lr': 0, 'momentum': 0.9, 'nesterov': True}, 'lr_sched': 'CyclicLR', 'lr_sched_params': {'base_lr': 6.14421e-6, 'max_lr': 1}} 


parameters for file "ctbData.py"
- 'default_format_path': 'path to training dataset'
- 
{'default_format_path': 'data/dialog-bAbI-tasks/dstc2/defaultFormat.train', 'batch_size': 2} 


parameters for Lightning Trainer 
- For a list of parameters, see Trainer.__init__(...) in PyTorch Lightning documentation
- The following will make Lightning fail because of a bug: 
     'auto_lr_find': True, 'auto_scale_batch_size': True
- 
{'gpus': 1, 'auto_lr_find': False, 'auto_scale_batch_size': False}
