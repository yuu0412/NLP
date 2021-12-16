import torch

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