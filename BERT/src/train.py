import pandas as pd
import numpy as np
import torch
import engine
import dataset
import config

from model import BERTBaseUncased
from sklearn.model_selection import train_test_split
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def train():
    df = pd.read_csv(config.TRAINING_FILE).fillna("none")
    df['sentiment'] = df['sentiment'].map(
                            {
                            "positive": 1,
                            "negative": 0
                            }
                        )

    df_train, df_valid = train_test_split(df,
                            test_size=0.1,
                            random_state=42,
                            stratify=df.sentiment.values
                        )

    # reset index of both splits
    df_train = df_train.reset_index(drop=True)                             
    df_valid = df_valid.reset_index(drop=True)                             

    train_dataset = dataset.BERTDataset(
        review=df_train.review.values,
        target=df_train.sentiment.values
    )

    train_dataloader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=config.TRAIN_BATCH_SIZE,
                            shuffle=False,
                            num_workers=4,
                        ) 

    valid_dataset = dataset.BERTDataset(
        review=df_valid.review.values,
        target=df_valid.sentiment.values
    )

    valid_dataloader = torch.utils.data.DataLoader(
                            valid_dataset,
                            batch_size=config.VALID_BATCH_SIZE,
                            shuffle=False,
                            num_workers=4,
                        )  

    device = torch.device("cuda")
    model = BERTBaseUncased()
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=int(len(df_train) / config.TRAIN_BATCH_SIZE) * config.EPOCHS
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(
                train_dataloader,
                model,
                optimizer,
                device,
                scheduler
            )
        outputs, targets = engine.eval_fn(
                valid_dataloader,
                model,
                device
            )        

        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(outputs, targets)
        print(f"Accuracy: {accuracy:.3f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), config.MODEL_PATH)

if __name__ == "__main__":
    train()