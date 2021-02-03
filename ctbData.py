'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
from logging import getLogger
from sys import exit
from typing import List, Dict, Union, Tuple
from pickle import load

logg = getLogger(__name__)


class ctbData(LightningDataModule):
    def __init__(self, d_params: dict):
        super().__init__()
        # Trainer('auto_scale_batch_size': True...) requires self.batch_size
        self.batch_size = d_params.pop('batch_size', 2)
        self.d_params = d_params

    def prepare_data(self,
                     tokenizer,
                     tokenizer_type: str,
                     no_training: bool = False,
                     no_testing: bool = False):
        self.tokenizer = tokenizer
        if tokenizer_type == "gpt2-dstc2":
            from utils.defaultFormat_to_gpt2Format import \
                    defaultFormat_to_gpt2Format
            data_info = defaultFormat_to_gpt2Format(
                self.tokenizer, tokenizer_type,
                self.d_params['default_format_path'])
            for name, f_path in data_info['f_paths'].items():
                with f_path.open('rb') as file:
                    if name == 'train' and not no_training:
                        self.train_data = ctbDataset(load(file))
                        logg.info(
                            f'{len(self.train_data)} examples in Training set')
                    elif name == 'valid' and not no_training:
                        self.valid_data = ctbDataset(load(file))
                        logg.info(
                            f'{len(self.valid_data)} examples in Valid set')
                    elif name == 'test' and not no_testing:
                        self.test_data = ctbDataset(load(file))
                        logg.info(
                            f'{len(self.test_data)} examples in Test set')
                    else:
                        assert (name == 'train' or name == 'valid'
                                or name == 'test')
        else:
            logg.critical(f'unknown tokenization: {tokenizer_type}')
            exit()

    def setup(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          sampler=RandomSampler(self.train_data),
                          batch_sampler=None,
                          num_workers=6,
                          collate_fn=self.gpt2_collater,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          sampler=RandomSampler(self.valid_data),
                          batch_sampler=None,
                          num_workers=6,
                          collate_fn=self.gpt2_collater,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          sampler=RandomSampler(self.test_data),
                          batch_sampler=None,
                          num_workers=6,
                          collate_fn=self.gpt2_collater,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0)

    def gpt2_collater(
            self,
            idxs_examples: List[Tuple[int,
                                      List[int]]]) -> Dict[str, torch.Tensor]:

        idxs, examples = zip(*idxs_examples)

        try:
            sep_idxs = [
                example.index(self.tokenizer.sep_token_id)
                for example in examples
            ]
        except ValueError:
            logg.critical('No sep_token in example')
            exit()

        example_lens = [len(example) for example in examples]
        max_example_len = max(example_lens)
        assert self.tokenizer.padding_side == 'right'

        input_ids = torch.LongTensor([
            (example + [self.tokenizer.pad_token_id] *
             (max_example_len - example_len))
            for example, example_len in zip(examples, example_lens)
        ])

        attention_mask = torch.FloatTensor([[1] * example_len + [0] *
                                            (max_example_len - example_len)
                                            for example_len in example_lens])

        token_type_ids = torch.LongTensor([[0] * sep_idx + [1] *
                                           (max_example_len - sep_idx)
                                           for sep_idx in sep_idxs])

        return {
            'model': {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            },
            'idxs': idxs
        }


class ctbDataset(Dataset):
    # example = feature plus label
    def __init__(self, features: Union[List[List[int]],
                                       List[Tuple[List[int], List[int]]]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self,
                    idx: int) -> Union[List[int], Tuple[List[int], List[int]]]:
        return (idx, self.features[idx])
