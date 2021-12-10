import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils.run_fold import run_fold
from utils.functions import init_logger
from models.BERTClassifier import BERTModel
from sklearn.model_selection import StratifiedKFold

from transformers import BertTokenizer

if __name__ == '__main__':
    
    class MedPaperTitleDataset(torch.utils.data.Dataset):
        def __init__(self, titles, labels=None, tokenizer=None):
            self.titles = titles.tolist()
            if labels is not None:
                self.labels = torch.tensor(labels.tolist(), dtype=torch.float)
            else:
                self.labels = None
                self.tokenizer = tokenizer
                self.encoder = self.tokenizer(self.titles, return_tensors='pt', padding=True, truncation=True)
                self.input_ids = self.encoder["input_ids"]
                self.attention_masks = self.encoder["attention_mask"]
        
        def __len__(self):
            return len(self.titles)

        def __getitem__(self, idx):
            input_id = self.input_ids[idx]
            attention_mask = self.attention_masks[idx]
            if self.labels is not None:
                label = self.labels[idx]
                return input_id, attention_mask, label
            else:
                return input_id, attention_mask

    df_train_title = pd.read_csv("input/train_title.csv")
    df_test_title = pd.read_csv("input/test_title.csv")
    train_title = df_train_title["title"]
    train_label = df_train_title["judgement"]
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for train_idx, val_idx in kf.split(train_title, train_label):
        train_x, val_x = train_title.iloc[train_idx], train_title.iloc[val_idx]
        train_y, val_y = train_label.iloc[train_idx], train_label.iloc[val_idx]

        border = sum(train_y==1) / sum(train_y==0)

        train_dataset = MedPaperTitleDataset(train_x, train_y, tokenizer)
        val_dataset = MedPaperTitleDataset(val_x, val_y,tokenizer)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

    model = BERTModel("bert-base-uncased")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    logger = init_logger("BERT_log")

    max_epochs = 10 # コンソールから入力できるようにする
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    run_fold(max_epochs, model, train_loader, val_loader, criterion, optimizer, device, logger)