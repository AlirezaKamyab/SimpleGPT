import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
from tokenizers import Tokenizer


class ParquetStreamDataset(IterableDataset):
    def __init__(
            self, 
            path:str, 
            column:str, 
            tokenizer:Tokenizer,
            batch_size:int=32, 
            context_window:int=256,
        ):
        super(ParquetStreamDataset, self).__init__()
        self.path = path
        self.column = column
        self.batch_size = batch_size
        self.context_window = context_window
        self.tokenizer = tokenizer
        self.EOS = self.tokenizer.token_to_id('[EOS]')

    def __iter__(self):
        parquet_file = pq.ParquetFile(self.path)
        all_tokens = []
        for batch in parquet_file.iter_batches(self.batch_size, columns=[self.column]):
            batch = batch.column(self.column).to_pylist()
            EOS = self.EOS
            for doc in batch:
                doc = self.tokenizer.encode(doc, add_special_tokens=False).ids
                all_tokens.extend(doc)
                all_tokens.append(EOS)
            
            while len(all_tokens) >= self.context_window:
                window = all_tokens[:self.context_window]
                all_tokens = all_tokens[self.context_window:]
                
                xi = window[:-1]
                yi = window[1:]
                yield {
                    'inputs': torch.tensor(xi, dtype=torch.long),
                    'labels': torch.tensor(yi, dtype=torch.long)
                }