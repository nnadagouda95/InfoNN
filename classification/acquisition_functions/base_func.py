# import libraries

import torch
import numpy as np

class BaseQuery:
    def __init__(self, model, device, labeled_loader, unlabeled_loader, args):  
        
        self.model = model
        self.device = device
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.args = args
    
    def query(self):
        pass
    
    def get_embedding(self):
    
        embeddings_l = []
        labels = []
        embeddings_u = []


        self.model.eval()
        with torch.no_grad():
            for data, target in self.labeled_loader:
                feat_l, _ = self.model(data.to(self.device, dtype=torch.float),1)
                feat_l = feat_l.squeeze(1)
                embeddings_l.append(feat_l)
                labels.append(target.to(self.device))
            for data, _ in self.unlabeled_loader:
                feat_u, _ = self.model(data.to(self.device, dtype=torch.float),1)
                feat_u = feat_u.squeeze(1)
                embeddings_u.append(feat_u)

        embeddings_l = torch.cat(embeddings_l)
        embeddings_u = torch.cat(embeddings_u)
        labels = torch.cat(labels).float()
        labels=labels.reshape(-1,1)

        return embeddings_u, embeddings_l, labels
