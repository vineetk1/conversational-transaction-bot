'''
Vineet Kumar, sioom.ai
List of new tokens that are not in tokenizer's vocabulary
'''

SPECIAL_TOKENS = {
    'bos_token': '<BOS>',
    'eos_token': '<EOS>',
    'unk_token': '<UNK>',
    'sep_token': '<SEP>',
    'pad_token': '<PAD>',
    # 'cls_token': '<CLS>',
    'mask_token': '<MASK>'
}

# ADDITIONAL_SPECIAL_TOKENS = ['<USER>', '<BOT>']

DSTC2_TOKENS = [
    '<SILENCE>', 'api_call', 'R_post_code', 'R_cuisine', 'R_location',
    'R_phone', 'R_address', 'R_price', 'R_rating'
]
