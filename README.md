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
python3 Main.py input_param_files/gpt2_dstc2
```
The user-settable hyper-parameters are in the file *input_param_files/gpt2_dstc2*. An explanation on the contents of this file is at *input_param_files/README.md*. A list of all the hyper-parameters is in the <a href="https://www.pytorchlightning.ai" target="_blank">PyTorch-Lightning documentation</a>, and any hyper-parameter can be used.    
To assist in Training, the two parameters *auto_lr_find* and *auto_scale_batch_size* in the file *input_param_files/gpt2_dstc2* enable the software to automatically find an initial Learning-Rate and a Batch-Size respectively.    
As training progresses, graphs of *"training-loss vs. epoch #"*, *"validation-loss vs. epoch #"*, and "learning-rate vs. batch #" are plotted in real-time on TensorBoard. Training is stopped by typing, at the Command-Line-Interface, the keystroke ctrl-c. The current training information is checkpointed, and training stops. Training can be resumed, at some future time, from the checkpointed file.   
Testing calculates the Perplexity of the model from the test dataset. A detailed statistics on the model is generated in the files *failed_dialogs_stat.txt* and *passed_dialogs_stat.txt*. The *failed_dialogs_stat.txt* file has information about the dialogs that failed, and *passed_dialogs_stat.txt* has information about the dialogs that passed.
## Resume training, validation, and testing a model with same hyper-parameters
Resume training a checkpoint model with the same model- and training-states by using the following command:
```
python3 Main.py input_param_files/gpt2_dstc2-resume_training
```
The user-settable hyper-parameters are in the file *input_param_files/gpt2_params-resume_training*.  An explanation on the contents of this file is at *input_param_files/README.md*.
## Change hyper-parameters and continue training, validation, and testing a model
Continue training a checkpoint model with the same model-state but different hyperparameters for the training-state by using the following command:
```
python3 Main.py input_param_files/gpt2_dstc2-load_change_params
```
The user-settable hyper-parameters are in the file *input_param_files/gpt2_dstc2-load_change_params*.  An explanation on the contents of this file is at *input_param_files/README.md*.
## Interact with the deployed model
Work In Progress.
## Fine-tuning Distilgpt2 with DSTC2 dataset
The Huggingface's Distilgpt2 transformer model has 6 layers, 768 dimensions and 12 heads, totaling 82M parameters. The DSTC2 dataset has 1618 dialogs (14404 examples) in the training set, 500 dialogs (4159 examples) in the valid set, and 1117 dialogs (11237 examples) in the test set.   
### &emsp; &emsp; Using NAG with varying initial Learning Rates
**Hyperparameters:**    
``*`` Optimizer Parameters -- SGD, lr: variable, momentum: 0.9, weight_decay: 0, dampening: 0, nesterov: True   
``*`` LR-Scheduler Parameters -- None   
``*`` Batch-size = 2 using Nvidia GTX 1080 GPU   
<img src=images/tensorboard,val_loss-5_epochs,nag.png width=800 height=500>     
*Graph: Validation-loss vs. Epoch for varying initial Learning-Rates.   
(Color of curve, Learning-Rate, Val-loss at epoch 0 and epoch 5) where en = euler number = 2.71828 -- (Aqua, en<sup>-12</sup>, 1.69, 0.8783), (Brown, en<sup>-11</sup>, 1.15, 0.6272), (Blue, en<sup>-10</sup>, 0.814, 0.4331), (Pink, en<sup>-6</sup>, 0.6802, 0.1893), (Orange, en<sup>-9</sup>, 0.6443, 0.2957), (Green, en<sup>-5</sup>, 0.6143, 0.1748), (Grey, en<sup>-8</sup>, 0.5248, 0.2281), (Green, en<sup>-7</sup>, 0.4954, 0.1887)*   
### &emsp; &emsp; Using Adam with varying initial Learning Rates
**Hyperparameters:**    
``*`` Optimizer Parameters -- Adam, lr: variable, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0, amsgrad: False   
``*`` LR-Scheduler Parameters -- None   
``*`` Batch-size = 2 using Nvidia GTX 1080 GPU   
<img src=images/tensorboard,val_loss-5_epochs,adam.png width=800 height=500>     
*Graph: Validation-loss vs. Epoch for varying initial Learning-Rates.   
(Color of curve, Learning-Rate, Val-loss at epoch 0 and epoch 5) where en = euler number = 2.71828 -- (Grey, en<sup>-13</sup>, 0.2225, 0.206), (Green, en<sup>-12</sup>, 0.3575, 0.1593), (Pink, en<sup>-11</sup>, 0.23, 0.1562), (Aqua, en<sup>-10</sup>, 0.1859, 0.1726), (Orange, en<sup>-7</sup>, 0.1814, 0.1971), (Blue, en<sup>-8</sup>, 0.1647, 0.2095), (Brown, en<sup>-9</sup>, 0.1634, 0.1924)*   
### &emsp; &emsp; Results
**Hyperparameters:**    
``*`` Optimizer Parameters -- Adam, lr: 1e-05, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0, amsgrad: False   
``*`` LR-Scheduler Parameters -- ReduceLROnPlateau, mode: min, patience: 6, factor: 0.1    
``*`` Batch-size = 2 using Nvidia GTX 1080 GPU   
<img src=images/train-train_loss_0.1683,val_loss_0.1581.png width=400 height=300> <img src=images/val-train_loss_0.1683,val_loss_0.1581.png width=400 height=300>   
&emsp; &emsp; &emsp;*Graph: Training-loss vs. Epoch.* &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;  *Graph: Validation-loss vs. Epoch.*    
**Test results:**   
Epoch 3 has the lowest validation loss of 0.1581 that is lower than the training loss of 0.1683. Running the test-set on an epoch 3 model, gives the following results:          
{'test_loss_step': 0.2913,    
&thinsp; 'test_loss_step_epoch': 0.3567,    
&thinsp; 'test_perplexity': 1.4287}     
**Exact match results:**   
Exact-match compares the label with the prediction. Suffice to say that this method produces the worst-case results because there are many possible correct predictions than just an exact-match with the labels.    
\*\* Number of turns = 11237   
&emsp;    \*\* Percent of turns with truncated inputs = (426/11237 x 100) = 3.79%   
&emsp;    \*\* Percent of turns that passed = (5142/11237 x 100) = 45.76%   
&emsp; &emsp;       \*\* Percent of turns that passed with truncated inputs = (195/426 x 100) = 45.77%   
&emsp; &emsp;       \*\* Percent of turns that passed with untruncated inputs = (4947/10811 x 100) = 45.76%   
&emsp;    \*\* Percent of turns that passed at each turn-number in dialogs -- (Turn # in dialogs: # of such turns that       
&emsp; &emsp; passed/total number of such turns x 100 = result) -- (1: 1117/1117 = 100.00%), (2: 43/1117 = 3.85%),    
&emsp; &emsp; (3: 245/1117 = 21.93%), (4: 362/1117 = 32.41%), (5: 417/1116 = 37.37%), (6: 503/1098 = 45.81%),   
&emsp; &emsp; (7: 520/1022 = 50.88%), (8: 491/864 = 56.83%), (9: 383/686 = 55.83%), (10: 302/523 = 57.74%),    
&emsp; &emsp; (11: 224/393 = 57.00%), (12: 173/287 = 60.28%), (13: 93/199 = 46.73%), (14: 76/149 = 51.01%),         
&emsp; &emsp; (15: 48/107 = 44.86%), (16: 29/77 = 37.66%), (17: 29/60 = 48.33%), (18: 24/50 = 48.00%),               
&emsp; &emsp; (19: 19/39 = 48.72%), (20: 13/28 = 46.43%), (21: 9/20 = 45.00%), (22: 6/14 = 42.86%), (23: 4/11 = 36.36%),      
&emsp; &emsp; (24: 2/8 = 25.00%), (25: 2/6 = 33.33%), (26: 4/5 = 80.00%), (27: 2/4 = 50.00%), (28: 1/2 = 50.00%),       
&emsp; &emsp; (29: 1/1 = 100.00%)     
\*\* Number of dialogs = 1117   
&emsp;    \*\* Percent of dialogs that passed= 1/1117 x 100 = 0.09%   
&emsp; &emsp; \*\* (# of turns in dialog: # of such dialogs that passed/total number of such dialogs x 100 = result) --    
&emsp; &emsp; &emsp; (4: 0/1 = 0.00%), (5: 0/18 = 0.00%), (6: 0/76 = 0.00%), (7: 1/158 = 0.63%), (8: 0/178 = 0.00%),       
&emsp; &emsp; &emsp; (9: 0/163 = 0.00%), (10: 0/130 = 0.00%), (11: 0/106 = 0.00%), (12: 0/88 = 0.00%), (13: 0/50 = 0.00%),   
&emsp; &emsp; &emsp; (14: 0/42 = 0.00%), (15: 0/30 = 0.00%), (16: 0/17 = 0.00%), (17: 0/10 = 0.00%), (18: 0/11 = 0.00%),               
&emsp; &emsp; &emsp; (19: 0/11 = 0.00%), (20: 0/8 = 0.00%), (21: 0/6 = 0.00%), (22: 0/3 = 0.00%), (23: 0/3 = 0.00%),          
&emsp; &emsp; &emsp; (24: 0/2 = 0.00%), (25: 0/1 = 0.00%), (26: 0/1 = 0.00%), (27: 0/2 = 0.00%), (28: 0/1 = 0.00%),       
&emsp; &emsp; &emsp; (29: 0/1 = 0.00%)    
&emsp; &emsp; \*\* (# of consecutive turns that passed, counting from beginning of dialog: # of occurrences of such    
&emsp; &emsp; &emsp; consecutive turns) -- (1: 1074), (2: 38), (3: 3), (4: 1), (6: 1)    
## Fine-tuning GPT2 with DSTC2 dataset
The Huggingface's GPT2 transformer model has 12 layers, and 124M parameters. The DSTC2 dataset has 1618 dialogs (14404 examples) in the training set, 500 dialogs (4159 examples) in the valid set, and 1117 dialogs (11237 examples) in the test set.  
### &emsp; &emsp; Results
**Hyperparameters:**    
``*`` Optimizer Parameters -- Adam, lr: 5e-06, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0, amsgrad: False   
``*`` LR-Scheduler Parameters -- ReduceLROnPlateau, mode: min, patience: 1, factor: 0.1    
``*`` Batch-size = 1 using Nvidia GTX 1080 GPU   
<img src=images/gpt2-train-val_loss_0.234,train_loss_0.2349.png width=400 height=300> <img src=images/gpt2-val-val_loss_0.234,train_loss_0.2349.png width=400 height=300>   
&emsp; &emsp; &emsp;*Graph: Training-loss vs. Epoch.* &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;  *Graph: Validation-loss vs. Epoch.*    
**Test results:**   
Epoch 3 has the lowest validation loss of 0.234 that is lower than the training loss of 0.2349. Running the test-set on an epoch 3 model, gives the following results:          
{'test_loss_step': 0.2866,    
&thinsp; 'test_loss_step_epoch': 0.4956,    
&thinsp; 'test_perplexity': 1.6414}     
**Exact match results:**   
Exact-match compares the label with the prediction. Suffice to say that this method produces the worst-case results because there are many possible correct predictions than just an exact-match with the labels.    
\*\* Number of turns = 11237   
&emsp;    \*\* Percent of turns with truncated inputs = (426/11237 x 100) = 3.79%   
&emsp;    \*\* Percent of turns that passed = (5583/11237 x 100) = 49.68%   
&emsp; &emsp;       \*\* Percent of turns that passed with truncated inputs = (197/426 x 100) = 46.24%   
&emsp; &emsp;       \*\* Percent of turns that passed with untruncated inputs = (5386/10811 x 100) = 49.82%   
&emsp;    \*\* Percent of turns that passed at each turn-number in dialogs -- (Turn # in dialogs: # of such turns that       
&emsp; &emsp; passed/total number of such turns x 100 = result) -- (1: 1117/1117 = 100.00%), (2: 115/1117 = 10.30%),    
&emsp; &emsp; (3: 265/1117 = 23.72%), (4: 392/1117 = 35.09%), (5: 494/1116 = 44.27%), (6: 580/1098 = 52.82%),   
&emsp; &emsp; (7: 570/1022 = 55.77%), (8: 506/864 = 58.56%), (9: 417/686 = 60.79%), (10: 317/523 = 60.61%),    
&emsp; &emsp; (11: 238/393 = 60.56%), (12: 181/287 = 63.07%), (13: 103/199 = 51.76%), (14: 75/149 = 50.34%),    
&emsp; &emsp; (15: 55/107 = 51.40%), (16: 34/77 = 44.16%), (17: 27/60 = 45.00%), (18: 26/50 = 52.00%),     
&emsp; &emsp; (19: 20/39 = 51.28%), (20: 15/28 = 53.57%), (21: 9/20 = 45.00%), (22: 6/14 = 42.86%), (23: 6/11 = 54.55%),    
&emsp; &emsp; (24: 3/8 = 37.50%), (25: 3/6 = 50.00%), (26: 4/5 = 80.00%), (27: 3/4 = 75.00%), (28: 1/2 = 50.00%),    
&emsp; &emsp; (29: 1/1 = 100.00%)     
\*\* Number of dialogs = 1117   
&emsp;    \*\* Percent of dialogs that passed= 22/1117 x 100 = 1.97%   
&emsp; &emsp; \*\* (# of turns in dialog: # of such dialogs that passed/total number of such dialogs x 100 = result) --    
&emsp; &emsp; &emsp; (4: 0/1 = 0.00%), (5: 5/18 = 27.78%), (6: 11/76 = 14.47%), (7: 6/158 = 3.80%), (8: 0/178 = 0.00%),    
&emsp; &emsp; &emsp; (9: 0/163 = 0.00%), (10: 0/130 = 0.00%), (11: 0/106 = 0.00%), (12: 0/88 = 0.00%), (13: 0/50 = 0.00%),    
&emsp; &emsp; &emsp; (14: 0/42 = 0.00%), (15: 0/30 = 0.00%), (16: 0/17 = 0.00%), (17: 0/10 = 0.00%), (18: 0/11 = 0.00%),    
&emsp; &emsp; &emsp; (19: 0/11 = 0.00%), (20: 0/8 = 0.00%), (21: 0/6 = 0.00%), (22: 0/3 = 0.00%), (23: 0/3 = 0.00%),     
&emsp; &emsp; &emsp; (24: 0/2 = 0.00%), (25: 0/1 = 0.00%), (26: 0/1 = 0.00%), (27: 0/2 = 0.00%), (28: 0/1 = 0.00%),     
&emsp; &emsp; &emsp; (29: 0/1 = 0.00%)    
&emsp; &emsp; \*\* (# of consecutive turns that passed, counting from beginning of dialog: # of occurrences of such    
&emsp; &emsp; &emsp; consecutive turns) -- (1: 1002), (2: 80), (3: 10), (4: 7), (5: 12), (6: 6)   
