'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningModule
import torch
from logging import getLogger
from sys import exit
from typing import Dict, List
from importlib import import_module
from pathlib import Path
import copy
import pickle

logg = getLogger(__name__)


class ctbModel(LightningModule):
    def __init__(self,
                 d_params: dict,
                 special_tokens: Dict[str, str] = None,
                 dstc2_tokens: List[str] = None):
        super().__init__()
        self.save_hyperparameters()
        # ctbModel.load_from_checkpoint(...) requires that d_params not change
        self.d_params = copy.deepcopy(d_params)
        # Trainer('auto_lr_find': True,...) requires self.lr
        self.lr = self.d_params['optz_params'].pop(
            'lr', None) if 'optz_params' in self.d_params else None
        self.model_type = self.d_params.pop('model_type', None)
        self.tokenizer_type = self.d_params.pop('tokenizer_type', None)

        if self.model_type == "distilgpt2-dstc2" or\
                self.model_type == "distilgpt2":
            from transformers import GPT2LMHeadModel
            self.model = GPT2LMHeadModel.from_pretrained('distilgpt2')
        else:
            logg.critical(f'unknown model_type: {self.model_type}')
            exit()

        if self.tokenizer_type == "gpt2-dstc2":
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
            _ = self.tokenizer.add_special_tokens(special_tokens)
            _ = self.tokenizer.add_tokens(dstc2_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            logg.critical(f'unknown tokenizer_type: {self.tokenizer_type}')
            exit()

    def get_tokenizer(self):
        return self.tokenizer

    def get_model_id(self):
        return {
            'model_type': self.model_type,
            'tokenizer_type': self.tokenizer_type,
        }

    def change_hperparams(self, d_params: dict):
        self.d_params = d_params
        self.lr = self.d_params['optz_params'].pop(
            'lr', None) if 'optz_params' in self.d_params else None

    def forward(self):
        logg.debug('')

    def training_step(self, batch: Dict[str, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        loss = self.run_model(batch['model'])
        # logger=True => TensorBoard; x-axis is always in steps=batches
        self.log('train_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=False)
        return loss

    def training_epoch_end(self,
                           training_step_outputs: List[Dict[str,
                                                            torch.Tensor]]):
        avg_loss = torch.stack([x['loss']
                                for x in training_step_outputs]).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('train_loss_epoch', avg_loss,
                                          self.current_epoch)

    def validation_step(self, batch: Dict[str, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        loss = self.run_model(batch['model'])
        # checkpoint-callback monitors epoch val_loss, so on_epoch=True
        self.log('val_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=False)
        return loss

    def validation_epoch_end(self, val_step_outputs: List[torch.Tensor]):
        avg_loss = torch.stack(val_step_outputs).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('val_loss_epoch', avg_loss,
                                          self.current_epoch)

    def test_step(self, batch: Dict[str, torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        loss = self.run_model(batch['model'])
        # checkpoint-callback monitors epoch val_loss, so on_epoch=True
        self.log('test_loss_step',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        try:
            if self.pass_fail_stat:
                self.pass_fail_stat_step(batch)
        except AttributeError:
            pass
        return loss

    def test_epoch_end(self, test_step_outputs: List[torch.Tensor]):
        avg_loss = torch.stack(test_step_outputs).mean()
        ppl = torch.exp(avg_loss)
        self.log('test_perplexity',
                 ppl,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        try:
            if self.pass_fail_stat:
                self.pass_fail_stat_end()
        except AttributeError:
            pass

    def run_model(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**batch, labels=batch["input_ids"])
        return outputs[0]  # mean of losses from each example in batch

    def configure_optimizers(self):
        if 'optz' in self.d_params and self.d_params['optz']:
            if 'optz_params' in self.d_params and self.d_params['optz_params']:
                if self.lr is not None:
                    optimizer = getattr(import_module('torch.optim'),
                                        self.d_params['optz'])(
                                            self.parameters(),
                                            lr=self.lr,
                                            **self.d_params['optz_params'])
                else:
                    optimizer = getattr(import_module('torch.optim'),
                                        self.d_params['optz'])(
                                            self.parameters(),
                                            **self.d_params['optz_params'])
            else:
                if self.lr is not None:
                    optimizer = getattr(import_module('torch.optim'),
                                        self.d_params['optz'])(
                                            self.parameters(), lr=self.lr)
                else:
                    optimizer = getattr(import_module('torch.optim'),
                                        self.d_params['optz'])(
                                            self.parameters())

        if 'lr_sched' in self.d_params and self.d_params['lr_sched']:
            if 'lr_sched_params' in self.d_params and self.d_params[
                    'lr_sched_params']:
                scheduler = getattr(import_module('torch.optim.lr_scheduler'),
                                    self.d_params['lr_sched'])(
                                        optimizer=optimizer,
                                        **self.d_params['lr_sched_params'])
            else:
                scheduler = getattr(
                    import_module('torch.optim.lr_scheduler'),
                    self.d_params['lr_sched'])(optimizer=optimizer)

        # If scheduler is specified then optimizer must be specified
        # If Trainer('resume_from_checkpoint',...), then optimizer and
        # scheduler may not be specified
        if 'optimizer' in locals() and 'scheduler' in locals():
            return {
                'optimizer':
                optimizer,
                'lr_scheduler':
                scheduler,
                'monitor':
                'val_loss'
                if self.d_params['lr_sched'] == 'ReduceLROnPlateau' else None
            }
        elif 'optimizer' in locals():
            return optimizer

    def clear_pass_fail_stat(self):
        self.pass_fail_stat = False

    def set_pass_fail_stat(self):
        self.pass_fail_stat = True
        stat_dir = Path.cwd().joinpath('statistics')
        stat_dir.mkdir(exist_ok=True)
        self.temp_file = stat_dir.joinpath('temp.txt')
        self.temp_file.touch()
        #self.temp_file.write_text('')   # empty the file
        self.dlgs_pass_noTrunc_file = stat_dir.joinpath('dlgs_pass_noTrunc.txt')
        self.dlgs_pass_noTrunc_file.touch()
        self.dlgs_pass_noTrunc_file.write_text('')   # empty the file
        self.dlgs_fail_noTrunc_file = stat_dir.joinpath('dlgs_fail_noTrunc.txt')
        self.dlgs_fail_noTrunc_file.touch()
        self.dlgs_fail_noTrunc_file.write_text('')   # empty the file
        self.dlgs_pass_trunc_file = stat_dir.joinpath('dlgs_pass_trunc.txt')
        self.dlgs_pass_trunc_file.touch()
        self.dlgs_pass_trunc_file.write_text('')   # empty the file
        self.dlgs_fail_trunc_file = stat_dir.joinpath('dlgs_fail_trunc.txt')
        self.dlgs_fail_trunc_file.touch()
        self.dlgs_fail_trunc_file.write_text('')   # empty the file
        self.dlgs_stat_file = stat_dir.joinpath('dlgs_stat.txt')
        self.dlgs_stat_file.touch()
        self.dlgs_stat_file.write_text('')   # empty the file
        self.dlgs_idxs_file = stat_dir.joinpath('dlgs_idxs.test')
        if not self.dlgs_idxs_file.exists():
            logg.critical(f'Following file does not exist: {self.dlgs_idxs_file}')
            exit()

    def pass_fail_stat_step(self, batch: Dict[str, torch.Tensor]):
        self.pass_fail_stat_end()
        return
        # batch['model']['input_ids'] = batch of
        #           (<BOS> + sequence + <SEP> + labels + <EOS> + <PAD>..<PAD>)
        # batch['model']['token_type_ids'] =
        #          batch of (ones at (<SEP> + labels + <EOS> + <PAD>..<PAD>))
        # batch['model']['attention_mask'] =
        #      batch of (ones at (<BOS> + sequence + <SEP> + labels + <EOS>))
        #  batch['idxs'] = batch of (indexes of dialog-turns)

        sep_idxs = torch.count_nonzero(
            batch['model']['token_type_ids'],
            dim=1).sub(batch['model']['input_ids'].shape[1]).mul(-1).tolist()
        eos_idxs = (torch.count_nonzero(batch['model']['attention_mask'],
                                        dim=1)).sub(1).tolist()
        sep_idx_max = max(sep_idxs)

        for i, sep_idx in enumerate(sep_idxs):
            batch['model']['attention_mask'][i, :sep_idx + 1] = 1
            batch['model']['attention_mask'][i, sep_idx + 1:] = 0

        model_kwargs = {
            'attention_mask':
            batch['model']['attention_mask'][:, :sep_idx_max + 1],
            'token_type_ids':
            batch['model']['token_type_ids'][:, :sep_idx_max + 1]
        }
        # outputs = batch of (['model']['input_ids'][:, :sep_idx_max+1] +
        # predicted labels)
        outputs = self.model.generate(
            # parameter = None => replace with self.config.parameter
            input_ids=batch['model']['input_ids'][:, :sep_idx_max + 1],
            max_length=self.tokenizer.max_model_input_sizes['distilgpt2'],
            min_length=1,
            do_sample=False,
            early_stopping=None,
            num_beams=2,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
            bad_words_ids=None,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            length_penalty=1.0,
            no_repeat_ngram_size=None,
            num_return_sequences=1,
            decoder_start_token_id=None,
            use_cache=None,
            # num_beam_groups=None,  # this parameter is not in called program
            # diversity_penalty=None,  # this parametr is not in called program
            prefix_allowed_tokens_fn=None,
            **model_kwargs)

        eos_idxs_in_predicted_labels = (outputs[:, sep_idx_max + 1:].eq(
            self.tokenizer.eos_token_id)).nonzero(
                as_tuple=False)[:, -1].add(sep_idx_max + 1).tolist()
        with self.temp_file.open('a') as file:
            for i, (sep_idx, eos_idx, eos_idx_in_predicted_label) in enumerate(
                    zip(sep_idxs, eos_idxs, eos_idxs_in_predicted_labels)):
                idx_dlg_turn = batch['idxs'][i]
                exact_match = batch['model']['input_ids'][
                    i, sep_idx + 1:eos_idx].equal(
                        outputs[i, sep_idx_max + 1:eos_idx_in_predicted_label])
                input_text = self.tokenizer.decode(outputs[i, 1:sep_idx])
                actual_label = self.tokenizer.decode(
                    batch['model']['input_ids'][i, sep_idx + 1:eos_idx])
                predicted_label = self.tokenizer.decode(
                    outputs[i, sep_idx_max + 1:eos_idx_in_predicted_label])
                # line written to disk: index of dialog turn /t exact_match /t
                # input text /t actual label text /t predicted label text
                strng = (f'{idx_dlg_turn}\t{exact_match}\t{input_text}\t'
                         f'{actual_label}\t{predicted_label}\n')
                file.write(strng)

    def pass_fail_stat_end(self):
        with self.dlgs_idxs_file.open('rb') as bfile:
            dlgs_idxs = pickle.load(bfile)
        #dlgs_turns = self.temp_file.read_text()
        with self.temp_file.open('r') as dlgs_turns:
            for dlg_turn in dlgs_turns:
                pass
