import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils.run_fold import run_fold
from utils.functions import init_logger
from models.BERTClassifier import BERTClassifier
from sklearn.model_selection import StratifiedKFold
from datasets.med_paper import MedPaperTitleDataset
from sklearn.metrics import fbeta_score

from transformers import BertTokenizer, AutoTokenizer
import sys
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="medical_paper")
def main(cfg):
    df_train_title = pd.read_csv(hydra.utils.get_original_cwd()+"/data/train_title.csv")
    df_test_title = pd.read_csv(hydra.utils.get_original_cwd()+"/data/test_title.csv")
    train_title = df_train_title["title"]
    train_label = df_train_title["judgement"]
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    for train_idx, val_idx in kf.split(train_title, train_label):
        train_x, val_x = train_title.iloc[train_idx], train_title.iloc[val_idx]
        train_y, val_y = train_label.iloc[train_idx], train_label.iloc[val_idx]

        border = sum(train_y==1) / sum(train_y==0)
        train_dataset = MedPaperTitleDataset(train_x, train_y, tokenizer)
        val_dataset = MedPaperTitleDataset(val_x, val_y,tokenizer)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_loader.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_loader.batch_size)

        model = BERTClassifier(cfg.model.name, cfg.model.num_labels)
        
        criterion = nn.__getattribute__(cfg.criterion.name)()
        optimizer = torch.optim.__getattribute__(cfg.optimizer.name)(model.parameters(), lr=cfg.optimizer.lr)
        eval_method = fbeta_score
        logger = init_logger("train_log")

        max_epochs = 10 # コンソールから入力できるようにする
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        run_fold(max_epochs, model, train_loader, val_loader, criterion, optimizer, eval_method, device, logger)

if __name__ == '__main__':
    main()