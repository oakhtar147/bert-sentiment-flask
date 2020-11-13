import torch.nn as nn
import torch
import config
from tqdm import tqdm

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for batch_idx, dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = dataset["ids"]
        token_type_ids = dataset["token_type_ids"]
        mask = dataset["mask"]
        targets = dataset["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad() # make sure grads update correctly
        outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()

    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for batch_idx, dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = dataset["ids"]
            token_type_ids = dataset["token_type_ids"]
            mask = dataset["mask"]
            targets = dataset["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids
            )

            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return final_outputs, final_targets




