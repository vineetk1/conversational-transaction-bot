# Conversational transaction bot
The Conversational Transaction Bot executes transactions on user commands. A user initiates a dialogue with a text query to a bot. The bot understands the user text, initiates execution of operations at the backend, and responds to the user in text. The dialogue continues until, normally, the user terminates the dialogue when its requests have been serviced by the bot. The implementation is based on Deep Learning Transformers.
## Requirements
* PyTorch version >= 1.7.1
* Python version >= 3.8.5
* PyTorch-Lightning version >= 1.2.1
* Huggingface Transformers version >= 4.3.3
* Tensorboard version >= 2.4.1
## Installation
```
pip3 install transformers
pip3 install pytorch-lightning
pip3 install tensorboard
git clone https://github.com/vineetk1/conversational-transaction-bot.git
cd conversational-transaction-bot
```
Note that the default directory is *conversational-transaction-bot*. Unless otherwise stated, all commands from the Command-Line-Interface must be delivered from the default directory.
## Download DSTC2 dataset
1. Go to https://fb-public.app.box.com/s/chnq60iivzv5uckpvj2n2vijlyepze6w 
1. Download *dialog-bAbI-tasks_1_.tgz* in directory *data*  

Verify that the current working directory is *data*.    
```
tar zxvf dialog-bAbI-tasks_1_.tgz
rm dialog-bAbI-tasks_1_.tgz
```
Verify that the DSTC2 dataset is in the directory *data/dialog-bAbI-tasks*.   
## Convert DSTC2 dataset to the default format
Convert dataset into a default format. An example of the default format is shown in the file *convert_to_default_formats/default_format_example.md*.   

Verify that the current working directory is the default directory. Following command converts the downloaded dataset to the default format, and saves it in the files - *defaultFormat.train, defaultFormat.valid, defaultFormat.test* - of the directory *data/dialog-bAbI-tasks/dstc2*:
```
python3 convert_to_default_formats/dstc2_to_defaultFormat.py
```
Note that the above program converts the DSTC2 dataset to the default format. A new conversion program will have to be implemented for a dataset that has a different format from that of the DSTC2 dataset. 
## Train, validate, and test a model
Following command trains a model, saves the last checkpoint plus checkpoints that have the lowest validation loss, runs the test dataset on the checkpointed model with the lowest validation loss, and outputs a Perplexity value of the model:
```
python3 ctbMain.py input_param_files/distilgpt2_dstc2
```
The user-settable hyper-parameters are in the file *input_param_files/distilgpt2_dstc2*. An explanation on the contents of this file is at *input_param_files/README.md*. A list of all the hyper-parameters is in the <a href="https://www.pytorchlightning.ai" target="_blank">PyTorch-Lightning documentation</a>, and any hyper-parameter can be used.    
To assist in Training, the two parameters *auto_lr_find* and *auto_scale_batch_size* in the file *input_param_files/distilgpt2_dstc2* enable the software to automatically find an initial Learning-Rate and a Batch-Size respectively.    
As training progresses, graphs of *"training-loss vs. epoch #"*, *"validation-loss vs. epoch #"*, and "learning-rate vs. batch #" are plotted in real-time on TensorBoard. Training is stopped by typing, at the Command-Line-Interface, the keystroke ctrl-c. The current training information is checkpointed, and training stops. Training can be resumed, at some future time, from the checkpointed file.   
Testing calculates the Perplexity of the model from the test dataset. A detailed statistics on the model is generated in the files *failed_dialogs_stat.txt* and *passed_dialogs_stat.txt*. The *failed_dialogs_stat.txt* file has information about the dialogs that failed, and *passed_dialogs_stat.txt* has information about the dialogs that passed.
## Resume training, validation, and testing a model with same hyper-parameters
Resume training a checkpoint model with the same model- and training-states by using the following command:
```
python3 ctbMain.py input_param_files/distilgpt2_dstc2-resume_training
```
The user-settable hyper-parameters are in the file *input_param_files/distilgpt2_params-resume_training*.  An explanation on the contents of this file is at *input_param_files/README.md*.
## Start training, validation, and testing a model with different hyper-parameters
Start training a checkpoint model with the same model-state but different hyperparameters for the training-state by using the following command:
```
python3 ctbMain.py input_param_files/distilgpt2_dstc2-load_change_params
```
The user-settable hyper-parameters are in the file *input_param_files/distilgpt2_dstc2-load_change_params*.  An explanation on the contents of this file is at *input_param_files/README.md*.
## Interact with the deployed model
Work In Progress.
## Fine-tuning Distilgpt2 with DSTC2 dataset
### &emsp; &emsp; Using NAG with varying initial Learning Rates
**Hyperparameters:**    
``*`` Optimizer Parameters -- SGD, lr: variable, momentum: 0.9, weight_decay: 0, dampening: 0, nesterov: True   
``*`` LR-Scheduler Parameters -- None   
<img src=images/tensorboard,val_loss-5_epochs,nag.png width=800 height=500>     
*Graph: Validation-loss vs. Epoch for varying initial Learning-Rates.   
(Color of curve, Learning-Rate, Val-loss at epoch 0 and epoch 5) where en = euler number = 2.71828 -- (Aqua, en<sup>-12</sup>, 1.69, 0.8783), (Brown, en<sup>-11</sup>, 1.15, 0.6272), (Blue, en<sup>-10</sup>, 0.814, 0.4331), (Pink, en<sup>-6</sup>, 0.6802, 0.1893), (Orange, en<sup>-9</sup>, 0.6443, 0.2957), (Green, en<sup>-5</sup>, 0.6143, 0.1748), (Grey, en<sup>-8</sup>, 0.5248, 0.2281), (Green, en<sup>-7</sup>, 0.4954, 0.1887)*   
### &emsp; &emsp; Using Adam with varying initial Learning Rates
**Hyperparameters:**    
``*`` Optimizer Parameters -- Adam, lr: variable, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0, amsgrad: False   
``*`` LR-Scheduler Parameters -- None   
<img src=images/tensorboard,val_loss-5_epochs,adam.png width=800 height=500>     
*Graph: Validation-loss vs. Epoch for varying initial Learning-Rates.   
(Color of curve, Learning-Rate, Val-loss at epoch 0 and epoch 5) where en = euler number = 2.71828 -- (Grey, en<sup>-13</sup>, 0.2225, 0.206), (Green, en<sup>-12</sup>, 0.3575, 0.1593), (Pink, en<sup>-11</sup>, 0.23, 0.1562), (Aqua, en<sup>-10</sup>, 0.1859, 0.1726), (Orange, en<sup>-7</sup>, 0.1814, 0.1971), (Blue, en<sup>-8</sup>, 0.1647, 0.2095), (Brown, en<sup>-9</sup>, 0.1634, 0.1924)*   
### &emsp; &emsp; Results
**Hyperparameters:**    
``*`` Optimizer Parameters -- Adam, lr: 1e-05, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0, amsgrad: False   
``*`` LR-Scheduler Parameters -- ReduceLROnPlateau, mode: min, patience: 6, factor: 0.1   
<img src=images/train:train_loss=0.1683,val_loss=0.1581.png width=400 height=300>   
<img src=images/val:train_loss=0.1683,val_loss=0.1581.png width=400 height=300>
