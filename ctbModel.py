'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningModule
import torch
import logging
import sys
from typing import Dict

logg = logging.getLogger(__name__)


class ctbModel(LightningModule):
    def __init__(self, args, len_tokenizer: int):
        logg.debug('')
        super().__init__()
        self.args = args
        if args.model == "gpt2":
            from transformers import GPT2LMHeadModel
            self.model = GPT2LMHeadModel.from_pretrained('distilgpt2')
            self.model.resize_token_embeddings(len_tokenizer)
        else:
            logg.critical(f'unknown model: {self.args.model}')
            sys.exit()

    def forward(self):
        logg.debug('')

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        logg.debug('')
        outputs = self.model(**batch, labels=batch["input_ids"])
        loss = outputs[0]  # mean of losses from each example in batch
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 logger=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        logg.debug('')
        avg_loss = torch.stack([x['loss']
                                for x in training_step_outputs]).mean()
        self.logger.experiment.add_scalar('train_loss_epoch', avg_loss,
                                          self.current_epoch)

    def validation_step(self, batch, batch_idx):
        logg.debug('')
        outputs = self.model(**batch, labels=batch["input_ids"])
        loss = outputs[0]  # mean of losses from each example in batch
        self.log('val_loss',
                 loss,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_epoch_end(self, val_step_outputs):
        logg.debug('')
        avg_loss = torch.stack([x for x in val_step_outputs]).mean()
        self.logger.experiment.add_scalar('val_loss_epoch', avg_loss,
                                          self.current_epoch)

    def test_step(self, batch, batch_idx):
        logg.debug('')
        '''
        batch, y = batch
        y_hat = self(batch)

        loss = F.cross_entropy(y_hat, y.long())
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()

        return {'test_loss': loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}
    '''

    def test_step_end(self, batch_parts):
        logg.debug('')

    def test_epoch_end(self, test_step_outputs):
        logg.debug('')
        '''
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc, 'step': self.current_epoch}

        return {'log': tensorboard_logs}
    '''

    def configure_optimizers(self):
        logg.debug('')
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=2e-05,
            eps=1e-08)
