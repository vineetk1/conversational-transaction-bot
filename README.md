# Conversational transaction bot
The Conversational Transaction Bot executes transactions on user commands. A user initiates a dialogue with a text query to a bot. The bot understands the user text, initiates execution of operations at the backend, and responds to the user in text. The dialogue continues until, normally, the user terminates the dialogue when its requests have been serviced by the bot. The implementation is based on Deep Learning Transformers.
## Requirements
* PyTorch version >= 1.6.0
* Python version >= 3.8.5
* PyTorch-Lightning version used is 1.1.0
* Huggingface Transformers version used is 4.0.1
* Tensorboard version used is 2.3.0
## Installation
```
pip3 install transformers
pip3 install pytorch-lightning
pip3 install tensorboard
git clone https://github.com/vineetk1/conversational-transaction-bot.git
cd conversational-transaction-bot
```
Note that the default directory is *conversational-transaction-bot*   

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
All datasets must be converted to the default format. An example of the default format is shown in the file *convert_to_default_formats/default_format_example.md*.   

Verify that the current working directory is the default directory. Following command-line converts the downloaded dataset to the default format, and saves it in the files - *defaultFormat.train, defaultFormat.valid, defaultFormat.test* - of the directory *data/dialog-bAbI-tasks/dstc2*:
```
python3 convert_to_default_formats/dstc2_to_defaultFormat.py
```
Note that the above program converts the DSTC2 dataset to the default format. A new conversion program will have to be written for a dataset that is different from the DSTC2 dataset. 
## Train a model
Verify that the current working directory is the default directory. Following command-line trains a model, saves checkpoints that have the lowest validation loss, runs the test dataset on the checkpointed model that has the lowest validation loss, and outputs a Perplexity value of the model.
```
python3 ctbMain.py input_param_files/distilgpt2_params
```
The user-settable hyper-parameters are in the file *input_param_files/distilgpt2_params*. It is envisioned that there will be many such files, in the *input_param_files* directory, each with their own unique set of hyperparameters. A list of all hyper-parameters can be found in the PyTorch-Lightning documentation https://www.pytorchlightning.ai/, and any hyper-parameter can be used.
## Evaluate the trained model
Verify that the current working directory is the default directory.
```
python3 dialog_generate.py --task dialog_task data-bin/dialog/task6 --path checkpoints/dialog/task6/checkpoint_best.pt --batch-size 32 --beam 5
```
The above command generates two files, namely, *failed_dialogs_stat.txt* and *passed_dialogs_stat.txt*. The *failed_dialogs_stat.txt* file has information about all the dialogs that failed, and *passed_dialogs_stat.txt* has information about all the dialogs that passed.  
