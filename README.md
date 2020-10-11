# Conversational transaction bot
The Conversational Transaction Bot executes transactions on user commands. A user initiates a dialog with a text query to a bot. The bot understands the user text, initiates execution of operations at the backend, and responds to the user in text. The dialog continues until, normally, the user terminates the dialog when its requests have been serviced by the bot. The implementation is based on Transformers.
## Requirements
* PyTorch version >= 1.6.0
* Python version >= 3.8.5
## Installation
```
pip3 install transformers
pip3 install pytorch-lightning
git clone https://github.com/vineetk1/conversational-transaction-bot.git
cd conversational-transaction-bot
```
## Download dataset
1. Go to https://fb-public.app.box.com/s/chnq60iivzv5uckpvj2n2vijlyepze6w 
1. Download *dialog-bAbI-tasks_1_.tgz* in directory *conversational-transaction-bot/data*  

Verify that the current working directory is *conversational-transaction-bot/data*.    
```
tar zxvf dialog-bAbI-tasks_1_.tgz
rm dialog-bAbI-tasks_1_.tgz
```
Verify that the dataset is in directory *conversational-transaction-bot/data/dialog-bAbI-tasks*.   
## Train the model
Verify that the current working directory is *conversational-transaction-bot*.
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --task dialog_task data-bin/dialog/task6 --arch dialog_lstm_model --save-dir checkpoints/dialog/task6 --max-tokens 8192 --required-batch-size-multiple 1 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer nag --lr-scheduler fixed --force-anneal 100 --lr 0.1 --clip-norm 0.1 --min-lr 2.47033e-200
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
