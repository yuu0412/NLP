def training(model, train_loader, max_epoch, criterion, device):

    loss_log = []
    score_log = []

    for epoch in range(max_epoch):
        losses = []
        print(f'============= epoch:{epoch} =============')
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
                print(f'{n}:{sum(losses) / len(losses)}')

            del labels, outputs ,attention_mask, input_ids

        loss_average = sum(losses) / len(losses)
        loss_log.append(loss_average)
        # すべての変換が終わってからf_betascoreを計算する
        score = fbeta_score(y_trues, y_preds, average="binary", beta=7.0)
        score_log.append(score)
        print(f'epoch_{epoch}:loss={loss_average} score={score}')

    return loss_log