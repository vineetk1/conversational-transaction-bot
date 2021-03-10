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
from utils.dialogs_info_out import DialogsInfoOut

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
        elif self.model_type == "gpt2-dstc2":
            from transformers import GPT2LMHeadModel
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
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
            if self.dlgs_statistics:
                self.dlgs_statistics_step(batch)
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
            if self.dlgs_statistics:
                self.dlgs_statistics_end()
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

    def clear_dlgs_statistics(self):
        self.dlgs_statistics = False

    def set_dlgs_statistics(self):
        self.dlgs_statistics = True
        stat_dir = Path.cwd().joinpath('statistics')
        stat_dir.mkdir(exist_ok=True)
        self.temp_file = stat_dir.joinpath('temp.txt')
        self.temp_file.touch()
        self.temp_file.write_text('')   # empty the file
        self.dlgs_meta_file = stat_dir.joinpath('dlgs_metadata.test')
        if not self.dlgs_meta_file.exists():
            logg.critical(
                f'Following file does not exist: {self.dlgs_meta_file}')
            exit()
        self.turns_meta_file = stat_dir.joinpath('turns_metadata.test')
        if not self.turns_meta_file.exists():
            logg.critical(
                f'Following file does not exist: {self.turns_meta_file}')
            exit()
        self.dlg_info_out = DialogsInfoOut()

    def dlgs_statistics_step(self, batch: Dict[str, torch.Tensor]):
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
                # line written to disk: index of turn /t exact_match /t
                # input text /t actual label text /t predicted label text
                strng = (f'{idx_dlg_turn}\t{exact_match}\t{input_text}\t'
                         f'{actual_label}\t{predicted_label}\n')
                file.write(strng)

    def dlgs_statistics_end(self):
        with self.dlgs_meta_file.open('rb') as dlgs_meta_file:
            dlgs_meta = pickle.load(dlgs_meta_file)
            # list of dictionary, where a dictionary has metadata of a dlg
            # dlgs_meta[0] => an example of a dlg at index 0
            # dlgs_meta[0]['lineno'] => (int) line # from the original dataset
            #                           file of start of dlg
            # dlgs_meta[0]['idx_first_trn'] => (int) index of first turn in dlg
            # dlgs_meta[-1] => content of last dlg is different
            # dlgs_meta[-1]['lineno'] => (int) line # of start of previous dlg
            # dlgs_meta[-1]['idx_first_trn'] => (int) index of last turn of
            #                                   previous dlg plus 1
        with self.turns_meta_file.open('rb') as turns_meta_file:
            turns_meta = pickle.load(turns_meta_file)
            # list of dictionary, where a dictionary has metadata of a turn
            # turns_meta[0] => an example turn at index 0
            # turns_meta[0]['u_str'] => (str) user string part of the input
            # turns_meta[0]['truncation'] => (Union[None, Tuple[str, str]]) if
            #       input is truncated, then this is a tupe of (untruncated
            #       part of string, truncated part of string) else None

        turns = [{
            'exact_match': False,
            'input': "",
            'act_out': "",
            'pred_out': ""
        } for _ in range(dlgs_meta[-1]['idx_first_trn'])]
        with self.temp_file.open('r') as turns_file:
            # sort in asending order of turns' indexes
            for turn_str in turns_file:
                turn_lst = turn_str.rstrip('\n').split('\t')
                turns[int(turn_lst[0])]['exact_match'] = turn_lst[1] == 'True'
                turns[int(turn_lst[0])]['input'] = copy.deepcopy(turn_lst[2])
                turns[int(turn_lst[0])]['act_out'] = copy.deepcopy(turn_lst[3])
                turns[int(turn_lst[0])]['pred_out'] = copy.deepcopy(
                    turn_lst[4])
                # turns[3] => an example of a turn at index 3
                # turns[3]['exact_match'] => (bool) True if actual_output ==
                #                            pedicted_output, else False
                # turns[3]['input'] => (str) input
                # turns[3]['act_out'] => (str) actual-output or label
                # turns[3]['pred_out'] => (str) predicted-output

        assert len(turns_meta) == len(turns)
        dlg_meta = dlgs_meta[0]
        for next_dlg_meta in dlgs_meta[1:]:
            # find if dialog passed
            dlg_passed = all([
                turn['exact_match'] for turn in
                turns[dlg_meta['idx_first_trn']:next_dlg_meta['idx_first_trn']]
            ])
            # find # of consecutive turns, counting from beginning, that passed
            for num_consec_turns_passed, turn in enumerate(turns[
                    dlg_meta['idx_first_trn']:next_dlg_meta['idx_first_trn']]):
                if not turn['exact_match']:
                    break

            self.dlg_info_out.dlg_meta(
                lineno=dlg_meta['lineno'],
                passed=dlg_passed,
                num_consec_turns_passed=num_consec_turns_passed,
                num_turns_in_dlg=next_dlg_meta['idx_first_trn'] -
                dlg_meta['idx_first_trn'])

            for i, (turn, turn_meta) in enumerate(
                    zip(
                        turns[dlg_meta['idx_first_trn']:
                              next_dlg_meta['idx_first_trn']],
                        turns_meta[dlg_meta['idx_first_trn']:
                                   next_dlg_meta['idx_first_trn']])):
                self.dlg_info_out.turn_meta(
                    dlg_passed=dlg_passed,
                    turn_num_in_dlgs=i + 1,
                    passed=turn['exact_match'],
                    truncation=turn_meta['truncation'],
                    user_inp=turn['input'][-len(turn_meta['u_str']):],
                    actual_output=turn['act_out'],
                    predicted_output=turn['pred_out'])
            dlg_meta = next_dlg_meta
        self.dlg_info_out.print_statistics()
        self.temp_file.unlink(missing_ok=False)
        pass
