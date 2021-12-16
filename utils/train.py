import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import fbeta_score

import numpy as np
import tqdm


from sklearn.metrics import fbeta_score

def training(model, train_loader, criterion, optimizer, eval_method, device, logger):

    losses = []
    model.train()

    for n, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        border = 0.023856671381423857
        y_true = labels.cpu().detach().numpy()
        y_pred = np.where(outputs.cpu().detach().numpy() < border, 0, 1)

        try:
            y_trues = np.concatenate([y_trues, y_true])
            y_preds = np.concatenate([y_preds, y_pred])
        except:
            y_preds = y_pred
            y_trues = y_true

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n % 100 == 0:
            logger.info(f'[{n}/{len(train_loader)}]: loss:{sum(losses) / len(losses)}')

        del labels, outputs ,attention_mask, input_ids

    loss_average = sum(losses) / len(losses)
    # すべての変換が終わってからfbeta_scoreを計算する
    score = eval_method(y_trues, y_preds, average="binary", beta=7.0)

    return loss_average, score