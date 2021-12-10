from sklearn.metrics import fbeta_score

def evaluation(model, val_loader, criterion, device):
    losses = []

    for n, batch in enumerate(val_loader):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        border = 0.023856671381423857
        y_true = labels.cpu().detach().numpy()
        y_pred = np.where(outputs.cpu().detach().numpy() < border, 0, 1)

        try:
            y_trues = np.concatenate([y_trues, y_true])
            y_preds = np.concatenate([y_preds, y_pred])
        except:
            y_preds = y_pred
            y_trues = y_true

    # すべての変換が終わってからf_betascoreを計算する
    score = fbeta_score(y_trues, y_preds, average="binary", beta=7.0)
    
    return losses, score