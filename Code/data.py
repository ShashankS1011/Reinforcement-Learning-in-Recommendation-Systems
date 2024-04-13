import os
import numpy as np
import pandas as pd
import torch
from model import ImpressionSimulator
from torch.utils.data import Dataset
from tqdm import tqdm
import dgl


class MovieLensDataset(Dataset):
    def __init__(self, filepath, device="cpu",encoding="latin-1"):
        self.device = device
        ratings, users, items = self.load_data(filepath)
         # Convert user data to appropriate numeric types
        for column in users.columns:
            if users[column].dtype == 'object':
                users[column] = pd.factorize(users[column])[0]
        G, labels, train_idx, val_idx, test_idx, train_size, val_size, test_size = self.bulid_ML_graph(ratings, users, items)


        self.user_feats = {}
        self.item_feats = {}
        self.impression_feats = {}
        print(device)
        self.user_feats["categorical_feats"] = torch.LongTensor(
            users.values).to(device)
        # Filter out non-numeric columns from items
        numeric_columns = items.select_dtypes(include=[np.number]).columns
        numeric_items = items[numeric_columns]
        # Convert to PyTorch tensor
        self.item_feats["categorical_feats"] = torch.LongTensor(numeric_items.values[:, :2])
        # Filter out non-numeric columns from items
        numeric_columns = items.select_dtypes(include=[np.number]).columns
        numeric_items = items[numeric_columns]

        # Convert to PyTorch tensor
        self.item_feats["real_feats"] = torch.FloatTensor(numeric_items.values[:, 2:])

        self.impression_feats["user_ids"] = torch.LongTensor(
            ratings.values[:, 0]).to(device)
        self.impression_feats["item_ids"] = torch.LongTensor(
            ratings.values[:, 1]).to(device)
        self.impression_feats["real_feats"] = torch.FloatTensor(
            ratings.values[:, 3]).view(-1, 1).to(device)
        self.impression_feats["labels"] = torch.FloatTensor(
            ratings.values[:, 2]).to(device)

        self.G = G
        self.labels=labels
        self.train_idx=train_idx
        self.val_idx=val_idx
        self.test_idx=test_idx
        self.train_size=train_size
        self.val_size=val_size
        self.test_size =test_size

    def __len__(self):
        return len(self.impression_feats["user_ids"])

    def __getitem__(self, idx):
        labels = self.impression_feats["labels"][idx]
        feats = {}
        feats["impression_feats"] = {}
        feats["impression_feats"]["real_feats"] = self.impression_feats[
            "real_feats"][idx]
        user_id = self.impression_feats["user_ids"][idx]
        item_id = self.impression_feats["item_ids"][idx]
        feats["user_feats"] = {
            key: value[user_id - 1]
            for key, value in self.user_feats.items()
        }
        feats["item_feats"] = {
            key: value[item_id - 1]
            for key, value in self.item_feats.items()
        }
        return feats, labels

    def load_data(self, filepath):
        names = "UserID::MovieID::Rating::Timestamp".split("::")
        ratings = pd.read_csv(
            os.path.join(filepath, "ratings.dat"),
            sep="::",
            names=names,
            engine="python",encoding="latin-1")
        ratings["Rating"] = (ratings["Rating"] > 3).astype(int)
        ratings["Timestamp"] = (
            ratings["Timestamp"] - ratings["Timestamp"].min()
        ) / float(ratings["Timestamp"].max() - ratings["Timestamp"].min())

        names = "UserID::Gender::Age::Occupation::Zip-code".split("::")
        users = pd.read_csv(
            os.path.join(filepath, "users.dat"),
            sep="::",
            names=names,
            engine="python",encoding="latin-1")
        for i in range(1, users.shape[1]):
            users.iloc[:, i] = pd.factorize(users.iloc[:, i])[0]

        names = "MovieID::Title::Genres".split("::")
        Genres = [
            "Action", "Adventure", "Animation", "Children's", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
            "Western"
        ]
        movies = pd.read_csv(
            os.path.join(filepath, "movies.dat"),
            sep="::",
            names=names,
            engine="python",encoding="ISO-8859-1")
        movies["Year"] = movies["Title"].apply(lambda x: x[-5:-1])
        for genre in Genres:
            movies[genre] = movies["Genres"].apply(lambda x: genre in x)
        movies.iloc[:, 3] = pd.factorize(movies.iloc[:, 3])[0]
        movies.iloc[:, 4:] = movies.iloc[:, 4:].astype(float)
        movies = movies.loc[:, ["MovieID", "Year"] + Genres]
        movies.iloc[:, 2:] = movies.iloc[:, 2:].div(
            movies.iloc[:, 2:].sum(axis=1), axis=0)

        movie_id_map = {}
        for i in range(movies.shape[0]):
            movie_id_map[movies.loc[i, "MovieID"]] = i + 1

        movies["MovieID"] = movies["MovieID"].apply(lambda x: movie_id_map[x])
        ratings["MovieID"] = ratings["MovieID"].apply(
            lambda x: movie_id_map[x])

        self.NUM_ITEMS = len(movies.MovieID.unique())
        self.NUM_YEARS = len(movies.Year.unique())
        self.NUM_GENRES = movies.shape[1] - 2

        self.NUM_USERS = len(users.UserID.unique())
        self.NUM_OCCUPS = len(users.Occupation.unique())
        self.NUM_AGES = len(users.Age.unique())
        self.NUM_ZIPS = len(users["Zip-code"].unique())

        return ratings, users, movies


    def bulid_ML_graph(self,ratings, users, movies):

        UvsM_data=ratings.astype({'UserID': 'category', 'MovieID': 'category'})
        UvsM_user_ids = torch.LongTensor(UvsM_data['UserID'].cat.codes.values)
        UvsM_movie_ids = torch.LongTensor(UvsM_data['MovieID'].cat.codes.values)

        #-------------------------------user profile---------------------------
        UvsG_data=users.astype({'UserID': 'category', 'Gender': 'category'})
        UvsG_user_ids = torch.LongTensor(UvsG_data['UserID'].cat.codes.values)
        UvsG_gender_ids = torch.LongTensor(UvsG_data['Gender'].cat.codes.values)


        UvsA_data=users.astype({'UserID': 'category', 'Age': 'category'})
        UvsA_user_ids = torch.LongTensor(UvsA_data['UserID'].cat.codes.values)
        UvsA_age_ids = torch.LongTensor(UvsA_data['Age'].cat.codes.values)


        UvsO_data=users.astype({'UserID': 'category', 'Occupation': 'category'})
        UvsO_user_ids = torch.LongTensor(UvsO_data['UserID'].cat.codes.values)
        UvsO_occupation_ids = torch.LongTensor(UvsO_data['Occupation'].cat.codes.values)


        UvsZ_data=users.astype({'UserID': 'category', 'Zip-code': 'category'})
        UvsZ_user_ids = torch.LongTensor(UvsZ_data['UserID'].cat.codes.values)
        UvsZ_zip_ids = torch.LongTensor(UvsZ_data['Zip-code'].cat.codes.values)

        #-------------------------------movie profile---------------------------

        MvsY_data=movies.astype({'MovieID': 'category', 'Year': 'category'})
        MvsY_movie_ids = torch.LongTensor(MvsY_data['MovieID'].cat.codes.values)
        MvsY_year_ids = torch.LongTensor(MvsY_data['Year'].cat.codes.values)

        # Build graph
        G = dgl.heterograph({
            ('user', 'watched', 'movie'): (UvsM_user_ids, UvsM_movie_ids),
            ('movie', 'watched-by', 'user'): (UvsM_movie_ids, UvsM_user_ids),
            
            ('user', 'is', 'gender'): (UvsG_user_ids, UvsG_gender_ids),
            ('gender', 'exist', 'user'): (UvsG_gender_ids, UvsG_user_ids),
            
            ('user', 'is-of', 'age'): (UvsA_user_ids, UvsA_age_ids),
            ('age', 'has', 'user'): (UvsA_age_ids, UvsA_user_ids),
            
            ('user', 'stay in', 'occupation'): (UvsO_user_ids, UvsO_occupation_ids),
            ('occupation', 'adopt', 'user'): (UvsO_occupation_ids, UvsO_user_ids),
            
            ('user', 'live in', 'zip-code'): (UvsZ_user_ids, UvsZ_zip_ids),
            ('zip-code', 'owns', 'user'): (UvsZ_zip_ids, UvsZ_user_ids),
            
            ('movie', 'pubilised in', 'year'): (MvsY_movie_ids, MvsY_year_ids),
            ('year', 'publishing', 'movie'): (MvsY_year_ids, MvsY_movie_ids)
            
        })

        labels = UvsM_movie_ids
        uid = UvsM_data['UserID'].cat.codes.values
        mid = UvsM_data['MovieID'].cat.codes.values
        shuffle = np.random.permutation(uid) #1000209
        #80%/10\%/10\%
        #60\%/20\%/20\%

        #600000 200000 200209
        train_idx = torch.tensor(shuffle[0:600000]).long()
        val_idx = torch.tensor(shuffle[600000:800000]).long()
        test_idx = torch.tensor(shuffle[800000:]).long()

        return G, labels, train_idx, val_idx, test_idx, len(train_idx), len(val_idx), len(test_idx)


class SyntheticMovieLensDataset(Dataset):
    def __init__(self, item_embedding, year_embedding, user_embedding,
     gender_embedding, age_embedding, occupation_embedding, zip_embedding, filepath, simulator_path, synthetic_data_path, cut=0.764506,
                 device="cuda:0"):
        self.device = device
        self.cut = cut
        self.simulator = None

        ratings, users, items = self.load_data(filepath)

        self.user_feats = {}
        self.item_feats = {}
        self.impression_feats = {}

        self.user_feats["categorical_feats"] = torch.LongTensor(users.values)
        self.item_feats["categorical_feats"] = torch.LongTensor(
            items.values[:, :2])
        self.item_feats["real_feats"] = torch.FloatTensor(items.values[:, 2:])

        if os.path.exists(synthetic_data_path):
            self.impression_feats = torch.load(synthetic_data_path)
            self.impression_feats["labels"] = (
                self.impression_feats["label_probs"] >= cut).to(
                    dtype=torch.float32)
            print("loaded impression_feats.pt")
        else:
            print("generating impression feats")
            self.simulator = ImpressionSimulator(item_embedding, year_embedding, user_embedding,
     gender_embedding, age_embedding, occupation_embedding, zip_embedding, use_impression_feats=True)
            self.simulator.load_state_dict(torch.load(simulator_path))
            self.simulator = self.simulator.to(device)

            impressions = self.get_full_impressions(ratings)

            self.impression_feats["user_ids"] = torch.LongTensor(
                impressions[:, 0])
            self.impression_feats["item_ids"] = torch.LongTensor(
                impressions[:, 1])
            self.impression_feats["real_feats"] = torch.FloatTensor(
                impressions[:, 2]).view(-1, 1)
            self.impression_feats["labels"] = torch.zeros_like(
                self.impression_feats["real_feats"])


            self.impression_feats["label_probs"] = self.generate_labels()
            self.impression_feats["labels"] = (
                self.impression_feats["label_probs"] >= cut).to(
                    dtype=torch.float32)

            torch.save(self.impression_feats, synthetic_data_path)
            print("saved impression_feats")

    def __len__(self):
        return len(self.impression_feats["user_ids"])

    def __getitem__(self, idx):
        labels = self.impression_feats["labels"][idx]
        feats = {}
        feats["impression_feats"] = {}
        feats["impression_feats"]["real_feats"] = self.impression_feats[
            "real_feats"][idx]
        user_id = self.impression_feats["user_ids"][idx]
        item_id = self.impression_feats["item_ids"][idx]
        feats["user_feats"] = {
            key: value[user_id - 1]
            for key, value in self.user_feats.items()
        }
        feats["item_feats"] = {
            key: value[item_id - 1]
            for key, value in self.item_feats.items()
        }
        return feats, labels

    def load_data(self, filepath):
        names = "UserID::MovieID::Rating::Timestamp".split("::")
        ratings = pd.read_csv(
            os.path.join(filepath, "ratings.dat"),
            sep="::",
            names=names,
            engine="python")
        ratings["Rating"] = (ratings["Rating"] > 3).astype(int)
        ratings["Timestamp"] = (
            ratings["Timestamp"] - ratings["Timestamp"].min()
        ) / float(ratings["Timestamp"].max() - ratings["Timestamp"].min())

        names = "UserID::Gender::Age::Occupation::Zip-code".split("::")
        users = pd.read_csv(
            os.path.join(filepath, "users.dat"),
            sep="::",
            names=names,
            engine="python")
        for i in range(1, users.shape[1]):
            users.iloc[:, i] = pd.factorize(users.iloc[:, i])[0]

        names = "MovieID::Title::Genres".split("::")
        Genres = [
            "Action", "Adventure", "Animation", "Children's", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
            "Western"
        ]
        movies = pd.read_csv(
            os.path.join(filepath, "movies.dat"),
            sep="::",
            names=names,
            engine="python")
        movies["Year"] = movies["Title"].apply(lambda x: x[-5:-1])
        for genre in Genres:
            movies[genre] = movies["Genres"].apply(lambda x: genre in x)
        movies.iloc[:, 3] = pd.factorize(movies.iloc[:, 3])[0]
        movies.iloc[:, 4:] = movies.iloc[:, 4:].astype(float)
        movies = movies.loc[:, ["MovieID", "Year"] + Genres]
        movies.iloc[:, 2:] = movies.iloc[:, 2:].div(
            movies.iloc[:, 2:].sum(axis=1), axis=0)

        movie_id_map = {}
        for i in range(movies.shape[0]):
            movie_id_map[movies.loc[i, "MovieID"]] = i + 1

        movies["MovieID"] = movies["MovieID"].apply(lambda x: movie_id_map[x])
        ratings["MovieID"] = ratings["MovieID"].apply(
            lambda x: movie_id_map[x])

        self.NUM_ITEMS = len(movies.MovieID.unique())
        self.NUM_YEARS = len(movies.Year.unique())
        self.NUM_GENRES = movies.shape[1] - 2

        self.NUM_USERS = len(users.UserID.unique())
        self.NUM_OCCUPS = len(users.Occupation.unique())
        self.NUM_AGES = len(users.Age.unique())
        self.NUM_ZIPS = len(users["Zip-code"].unique())

        return ratings, users, movies

    def get_full_impressions(self, ratings):
        """Gets NUM_USERS x NUM_ITEMS impression features by iterating the user and item ids.
        The impression-level feature, i.e. the timestamp, is sampled from a normal distribution with
        mean and std as the empirical mean and std of each user's recorded timestamps in the real data.
        """
        timestamps = {}
        for i in range(len(ratings)):
            u_id = ratings.loc[i, "UserID"]
            timestamps[u_id] = timestamps.get(u_id, [])
            timestamps[u_id].append(ratings.loc[i, "Timestamp"])
        rs = np.random.RandomState(0)
        t_samples = []
        for i in range(self.NUM_USERS):
            u_id = i + 1
            t_samples.append(
                rs.normal(
                    loc=np.mean(timestamps[u_id]),
                    scale=np.std(timestamps[u_id]),
                    size=(self.NUM_ITEMS, )))
        t_samples = np.array(t_samples)

        impressions = []
        for i in range(self.NUM_USERS):
            for j in range(self.NUM_ITEMS):
                impressions.append([i + 1, j + 1, t_samples[i, j]])
        impressions = np.array(impressions)
        return impressions

    def to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, dict):
            transformed_data = {}
            for key in data:
                transformed_data[key] = self.to_device(data[key])
        elif type(data) == list:
            transformed_data = []
            for x in data:
                transformed_data.append(self.to_device(x))
        else:
            raise NotImplementedError(
                "Type {} not supported.".format(type(data)))
        return transformed_data

    def generate_labels(self):
        """Generates the binary labels using the simulator on every user-item pair."""
        with torch.no_grad():
            self.simulator.eval()
            preds = []
            for i in tqdm(range(len(self.impression_feats["labels"]) // 500)):
                feats, _ = self.__getitem__(
                    list(range(i * 500, (i + 1) * 500)))
                feats = self.to_device(feats)
                outputs = torch.sigmoid(self.simulator(**feats))
                preds += list(outputs.squeeze().cpu().numpy())
            if (i + 1) * 500 < len(self.impression_feats["labels"]):
                feats, _ = self.__getitem__(
                    list(
                        range((i + 1) * 500,
                              len(self.impression_feats["labels"]))))
                feats = self.to_device(feats)
                outputs = torch.sigmoid(self.simulator(**feats))
                preds += list(outputs.squeeze().cpu().numpy())
        return torch.FloatTensor(np.array(preds))