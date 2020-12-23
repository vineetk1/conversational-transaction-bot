'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningModule
import torch
from logging import getLogger
from sys import exit
from typing import Dict, List

logg = getLogger(__name__)


class ctbModel(LightningModule):
    def __init__(self,
                 d_params: dict,
                 special_tokens: Dict[str, str] = None,
                 dstc2_tokens: List[str] = None):
        logg.debug('')
        super().__init__()
        self.save_hyperparameters()
        self.model_type = d_params['model_type']
        self.tokenizer_type = d_params['tokenizer_type']
        if self.model_type == "distilgpt2-dstc2":
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

    def forward(self):
        logg.debug('')

    def training_step(self, batch: Dict[str, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        logg.debug('')
        loss = self.run_model(batch)
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
        logg.debug('')
        avg_loss = torch.stack([x['loss']
                                for x in training_step_outputs]).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('train_loss_epoch', avg_loss,
                                          self.current_epoch)

    def validation_step(self, batch: Dict[str, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        logg.debug('')
        loss = self.run_model(batch)
        # checkpoint-callback monitors epoch val_loss, so on_epoch=True
        self.log('val_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=False)
        return loss

    def validation_epoch_end(self, val_step_outputs: List[torch.Tensor]):
        logg.debug('')
        avg_loss = torch.stack([x for x in val_step_outputs]).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('val_loss_epoch', avg_loss,
                                          self.current_epoch)

    def test_step(self, batch: Dict[str, torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        logg.debug('')
        model_kwargs = {
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids']
        }

        # outputs = batch['input_ids'] + (batch of generated outputs)
        # batch['input_ids']=batch of (<BOS> + sequence + <SEP> + <PAD>..<PAD>)
        outputs = self.model.generate(
            # parameter = None => replace with self.config.parameter
            input_ids=batch['input_ids'],
            max_length=self.tokenizer.max_model_input_sizes['distilgpt2'],
            min_length=None,
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

        print(f"input_ids={self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)}")
        print(f"labels_ids={self.tokenizer.batch_decode(batch['label_ids'], skip_special_tokens=True)}")
        print(f"outputs={self.tokenizer.batch_decode(outputs[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)}")
        print('end')

        # checkpoint-callback monitors epoch val_loss, so on_epoch=True
        '''
        self.log('test_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss
        '''
        '''
        batch, y = batch
        y_hat = self(batch)

        loss = F.cross_entropy(y_hat, y.long())
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()

        return {'test_loss': loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}
    '''

    #def test_step_end(self, batch_parts):
    #    logg.debug('')

    def test_epoch_end(self, test_step_outputs: List[torch.Tensor]):
        logg.debug('')
        avg_loss = torch.stack([x for x in test_step_outputs]).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('test_loss_epoch', avg_loss,
                                          self.current_epoch)
        '''
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc, 'step': self.current_epoch}

        return {'log': tensorboard_logs}
    '''

    def run_model(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**batch, labels=batch["input_ids"])
        return outputs[0]  # mean of losses from each example in batch

    def configure_optimizers(self):
        logg.debug('')
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=2e-05,
            eps=1e-08)
