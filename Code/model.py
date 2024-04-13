import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ITEMS = 3883
NUM_YEARS = 81
NUM_GENRES = 18
NUM_USERS = 6040
NUM_OCCUPS = 21
NUM_AGES = 7
NUM_ZIPS = 3439

embedding_size=128


class ItemRep(nn.Module):
    """Item representation layer."""

    def __init__(self, item_embedding, year_embedding , emb_size=embedding_size):

        super(ItemRep, self).__init__()
        self.item_embedding = nn.Embedding(NUM_ITEMS + 1, emb_size)
        self.item_embedding = self.item_embedding.from_pretrained(item_embedding,freeze=emb_size)
        self.year_embedding = nn.Embedding(NUM_YEARS, emb_size)
        self.year_embedding = self.year_embedding.from_pretrained(year_embedding,freeze=emb_size)
        self.rep_dim = emb_size*2

    def forward(self, categorical_feats,real_feats=None):

        out = torch.cat(
            [
                self.item_embedding(categorical_feats[:, 0]-1),
                self.year_embedding(categorical_feats[:, 1]),
                # self.genre_linear(real_feats)
            ],
            dim=1)
        return out


class UserRep(nn.Module):
    """User representation layer."""

    def __init__(self, user_embedding, gender_embedding,age_embedding, occupation_embedding, zip_embedding,emb_size=embedding_size):

        super(UserRep, self).__init__()
        self.user_embedding = nn.Embedding(NUM_USERS + 1,emb_size)
        self.user_embedding = self.user_embedding.from_pretrained(user_embedding,freeze=emb_size)

        self.gender_embedding = nn.Embedding(2, emb_size)
        self.gender_embedding = self.gender_embedding.from_pretrained(gender_embedding,freeze=emb_size)

        self.age_embedding = nn.Embedding(NUM_AGES, emb_size)
        self.age_embedding = self.age_embedding.from_pretrained(age_embedding,freeze=emb_size)

        self.occup_embedding = nn.Embedding(NUM_OCCUPS, emb_size)
        self.occup_embedding = self.occup_embedding.from_pretrained(occupation_embedding,freeze=emb_size)

        self.zip_embedding = nn.Embedding(NUM_ZIPS, emb_size)
        self.zip_embedding = self.zip_embedding.from_pretrained(zip_embedding,freeze=emb_size)

        self.rep_dim = emb_size*5


    def forward(self, categorical_feats, real_feats=None):
        reps = [
            self.user_embedding(categorical_feats[:, 0]-1),
            self.gender_embedding(categorical_feats[:, 1]),
            self.age_embedding(categorical_feats[:, 2]),
            self.occup_embedding(categorical_feats[:, 3]),
            self.zip_embedding(categorical_feats[:, 4])
        ]
        out = torch.cat(reps, dim=1)
        return out


class ImpressionSimulator(nn.Module):
    """Simulator model that predicts the outcome of impression."""

    def __init__(self, item_embedding, 
        year_embedding, user_embedding, gender_embedding,age_embedding, 
        occupation_embedding, zip_embedding, hidden=100, use_impression_feats=False):


        super(ImpressionSimulator, self).__init__()
        self.user_rep = UserRep(user_embedding, gender_embedding,age_embedding, occupation_embedding, zip_embedding)
        self.item_rep = ItemRep(item_embedding, year_embedding)

        self.use_impression_feats = use_impression_feats
        input_dim = self.user_rep.rep_dim + self.item_rep.rep_dim

        if use_impression_feats:
            input_dim += 1
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(), nn.Linear(hidden, 50), nn.ReLU(), nn.Linear(50, 1))

    def forward(self, user_feats, item_feats, impression_feats=None):
        users = self.user_rep(**user_feats)

        items = self.item_rep(**item_feats)

        inputs = torch.cat([users, items], dim=1)
        if self.use_impression_feats:
            inputs = torch.cat([inputs, impression_feats["real_feats"]], dim=1)
        return self.linear(inputs).squeeze()


class Nominator(nn.Module):
    """Two tower nominator model."""

    def __init__(self, item_embedding, 
        year_embedding, user_embedding, gender_embedding,age_embedding, 
        occupation_embedding, zip_embedding, hidden=100, use_impression_feats=False):

        super(Nominator, self).__init__()
        self.user_rep = UserRep(user_embedding, gender_embedding,age_embedding, occupation_embedding, zip_embedding)
        self.item_rep = ItemRep(item_embedding, year_embedding)
        self.linear = nn.Linear(self.user_rep.rep_dim, self.item_rep.rep_dim)
        self.binary = True

    def forward(self, user_feats, item_feats):
        users = self.linear(F.relu(self.user_rep(**user_feats)))
        users = torch.unsqueeze(users, 2)  # (b, h) -> (b, h, 1)
        items = self.item_rep(**item_feats)
        if self.binary:
            items = torch.unsqueeze(items, 1)  # (b, h) -> (b, 1, h)
        else:
            items = torch.unsqueeze(items, 0).expand(users.size(0), -1,
                                                     -1)  # (c, h) -> (b, c, h)
        logits = torch.bmm(items, users).squeeze()
        return logits

    def set_binary(self, binary=True):
        self.binary = binary


class Ranker(nn.Module):
    """Ranker model."""

    def __init__(self, item_embedding, 
        year_embedding, user_embedding, gender_embedding,age_embedding, 
        occupation_embedding, zip_embedding, hidden=100, use_impression_feats=False):
    
        super(Ranker, self).__init__()
        self.user_rep = UserRep(user_embedding, gender_embedding,age_embedding, occupation_embedding, zip_embedding)
        self.item_rep = ItemRep(item_embedding, year_embedding)
        self.linear = nn.Linear(self.user_rep.rep_dim + 1,
                                self.item_rep.rep_dim)
        self.binary = True

    def forward(self, user_feats, item_feats, impression_feats):
        users = self.user_rep(**user_feats)
        context_users = torch.cat(
            [users, impression_feats["real_feats"]], dim=1)
        context_users = self.linear(context_users)
        context_users = torch.unsqueeze(context_users,
                                        2)  # (b, h) -> (b, h, 1)
        items = self.item_rep(**item_feats)
        if self.binary:
            items = torch.unsqueeze(items, 1)  # (b, h) -> (b, 1, h)
        else:
            items = torch.unsqueeze(items, 0).expand(
                users.size(0), -1, -1)  # (c, h) -> (b, c, h), c=#items
        logits = torch.bmm(items, context_users).squeeze()
        return logits

    def set_binary(self, binary=True):
        self.binary = binary
