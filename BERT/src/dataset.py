from torch._C import dtype
import config
import torch


class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())      

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len
        )

        ids = inputs["input_ids"] # word2idx
        mask = inputs["attention_mask"] # padded tokens that attention mechanism does not attend to
        token_type_ids = inputs["token_type_ids"] # differentiate between to sequences, 1 since using one sentence in tokenizer

        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            "ids": torch.tensor(ids, dtype=torch.int64),
            "mask": torch.tensor(mask, dtype=torch.int64),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.int64),
            "targets": torch.tensor(self.target[item], dtype=torch.float)
        }


