# Conversational transaction bot
The Conversational Transaction Bot executes transactions on user commands. A user initiates a dialogue with a text query to a bot. The bot understands the user text, initiates execution of operations at the backend, and responds to the user in text. The dialogue continues until, normally, the user terminates the dialogue when its requests have been serviced by the bot. The implementation is based on Deep Learning Transformers.
## Requirements
* PyTorch version >= 1.6.0
* Python version >= 3.8.5
* PyTorch-Lightning version used is 1.0.8
* Huggingface Transformers version used is 3.3.1
## Installation
```
pip3 install transformers
pip3 install pytorch-lightning
git clone https://github.com/vineetk1/conversational-transaction-bot.git
cd conversational-transaction-bot
```
Note that the default directory is *conversational-transaction-bot*   

## Download DSTC2 dataset
1. Go to https://fb-public.app.box.com/s/chnq60iivzv5uckpvj2n2vijlyepze6w 
1. Download *dialog-bAbI-tasks_1_.tgz* in directory *data*  
1. Verify that the dataset is in directory *data/dialog-bAbI-tasks*   

Verify that the current working directory is *conversational-transaction-bot/data*.    
```
tar zxvf dialog-bAbI-tasks_1_.tgz
rm dialog-bAbI-tasks_1_.tgz
```
Verify that the dataset is in the directory *conversational-transaction-bot/data/dialog-bAbI-tasks*.   
## Convert DSTC2 dataset to the default format
All datasets must be converted to the default format. An example of the default format is shown in the file *conversational-transaction-bot/convert_to_default_formats/default_format_example.md*.   

Verify that the current working directory is *conversational-transaction-bot*. Following command converts the downloaded dataset to the default format in the files - *defaultFormat.train, defaultFormat.valid, defaultFormat.test* - of the directory *conversational-transaction-bot/data/dialog-bAbI-tasks/dstc2*:
```
python3 convert_to_default_formats/dstc2_to_defaultFormat.py
```
Note that the above program converts the DSTC2 dataset to the default format. A new conversion program will have to be written for a dataset that is different from the DSTC2 dataset. 
## Train the model
Verify that the current working directory is *conversational-transaction-bot*.
```
python3 train.py --gpus 1 --deterministic True --model gpt2 --tokenizer gpt2 --default_format_path data/dialog-bAbI-tasks/dstc2/defaultFormat.train
```
**NOTE:** If a model has previously been trained then it is in the file *checkpoints/dialog/task6/checkpoint_best.pt*   
If training again to generate a new model then the previous obsolete model must be removed, otherwise training will resume from the last best checkpoint model. A brute-force method to remove the obsolete model is to remove the directory *fairseq/checkpoints* as follows:
```
rm -r checkpoints
```
## Evaluate the trained model
Verify that the current working directory is *conversational-transaction-bot*.
```
python3 dialog_generate.py --task dialog_task data-bin/dialog/task6 --path checkpoints/dialog/task6/checkpoint_best.pt --batch-size 32 --beam 5
```
The above command generates two files, namely, *failed_dialogs_stat.txt* and *passed_dialogs_stat.txt*. The *failed_dialogs_stat.txt* file has information about all the dialogs that failed, and *passed_dialogs_stat.txt* has information about all the dialogs that passed.  
