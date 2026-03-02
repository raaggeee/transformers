import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq = seq

        self.sos = torch.tensor([self.tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos = torch.tensor([self.tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad = torch.tensor([self.tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)    
    
    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        src_text = src_tgt_pair["transaltion"][self.src_lang]
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]
        
        #extract the encoded ids of src and tgt text
        encoder_input_tokens = self.tokenizer_src(src_text).ids
        decoder_input_tokens = self.tokenizer_tgt(tgt_text).ids 

        enc_num_padding_tokens = self.seq - len(encoder_input_tokens) - 2 # for adding sos and eos
        dec_num_padding_tokens = self.seq - len(decoder_input_tokens) - 1 # only add eos #eos will be added with label

        encoder_token = torch.cat(
            [self.sos,
            torch.tensor(encoder_input_tokens, dtype=torch.int64),
            self.eos,
            torch.tensor([self.pad * enc_num_padding_tokens], dtype=torch.int64)],
            dim=0
        )

        decoder_token = torch.cat(
            [
                self.sos,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad * dec_num_padding_tokens], dtype=torch.int64)

            ],
            dim=0
        )

        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eof,
                torch.tensor([self.pad * dec_num_padding_tokens], dtype=torch.int64)

            ]
        )

        return {
            "encoder_input": encoder_token,
            "decoder_input": decoder_token,
            "encoder_mask": (encoder_token != self.pad).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_token != self.pad).unsqueeze(0).int & casual_mask(decoder_token.size(0))
        }
    
def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size), diagonal=1).type(torch.int))
    return mask == 0
