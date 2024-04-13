import scipy.io
from scipy import io as sio
import urllib.request
import dgl
import math
import numpy as np
from MGALayer import *
import argparse
from data import MovieLensDataset
import pandas as pd
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings
import dgl
warnings.filterwarnings('ignore')

torch.manual_seed(0)



class MGAEmbedding:
    def __init__(self, args, filepath, device="cuda"):
        self.args=args
        dataset = MovieLensDataset(filepath, device=device)

        if self.args.dataset == "ML":
        	self.G = dataset.G 
        	self.labels = dataset.labels
        	self.train_idx = dataset.train_idx
        	self.val_idx = dataset.val_idx
        	self.test_idx = dataset.test_idx
        	self.train_size = dataset.train_size
        	self.val_size = dataset.val_size
        	self.test_size = dataset.test_size 


        else:
        	raise NotImplementedError("{} not supported.".format(self.args.dataset))
        print('Successfully bulid hetergenous graph for {}'.format(self.args.dataset))
        print('Training/validation/test size', self.train_size, self.val_size, self.test_size)
        print(self.G)

        self.emb=self.Train_MGA()


    

    def forward(self, model, G,device="cuda"):
        best_val_acc = torch.tensor(0)
        best_test_acc = torch.tensor(0)
        train_step = torch.tensor(0)
        for epoch in np.arange(self.args.n_epoch) + 1:
            model.train()
            train_step += 1
            for train_idx in tqdm(self.train_loader):
                # The loss is computed only for labeled nodes.
                logits = model(G, 'user')
                loss = F.cross_entropy(logits[train_idx], self.labels[train_idx].to(device))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                self.optimizer.step()
                self.scheduler.step(train_step)
            if epoch % 5 == 0:
                model.eval()
                logits = model(G, 'user')
                pred   = logits.argmax(1).cpu()
                train_acc = (pred[train_idx] == self.labels[train_idx]).float().mean()
                val_acc   = (pred[self.val_idx]   == self.labels[self.val_idx]).float().mean()
                test_acc  = (pred[self.test_idx]  == self.labels[self.test_idx]).float().mean()
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                print('Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                    epoch,
                    self.optimizer.param_groups[0]['lr'], 
                    loss.item(),
                    train_acc.item(),
                    val_acc.item(),
                    best_val_acc.item(),
                    test_acc.item(),
                    best_test_acc.item(),
                ))

    def Train_MGA(self,device="cuda"):
        self.train_IDX = Batchwise(self.train_idx)
        self.train_loader = DataLoader(
        dataset=self.train_IDX,
        batch_size=256,
        num_workers=8)

        node_dict = {}
        edge_dict = {}
        for ntype in self.G.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in self.G.etypes:
            edge_dict[etype] = len(edge_dict)
            self.G.edges[etype].data['id'] = torch.ones(self.G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 

        #     Random initialize input feature
        for ntype in self.G.ntypes:
            emb = nn.Parameter(torch.Tensor(self.G.number_of_nodes(ntype), self.args.n_inp), requires_grad = False)
            nn.init.xavier_uniform_(emb)
            self.G.nodes[ntype].data['inp'] = emb

        self.G = self.G.to(device)

        model = MGA(self.G,
                    node_dict, edge_dict,
                    n_inp=self.args.n_inp,
                    n_hid=self.args.n_hid,
                    n_out=self.labels.max().item()+1,
                    n_layers=2,
                    n_heads=4,
                    use_norm = True).to(device)
        self.optimizer = torch.optim.AdamW(model.parameters())
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, total_steps=self.args.n_epoch, max_lr = self.args.max_lr)
        print('Training MGA ')

        self.forward(model, self.G)
        print('Training Stop')
        
        # return self.G.nodes



class Batchwise(Dataset):
    def __init__(self, inp_data):
        self.inp_data = inp_data
    def __getitem__(self, index):
        outputs = self.inp_data[index]
        return outputs
    def __len__(self):
        return len(self.inp_data)