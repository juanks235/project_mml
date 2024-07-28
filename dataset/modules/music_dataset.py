import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TorchMusicDataset(Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.examples = [
            os.path.join(self.dataset_folder, example_file)
            for example_file in os.listdir(self.dataset_folder)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = np.load(self.examples[index], allow_pickle=True).tolist()
        print(example.keys())
        return {
            "song_id": torch.tensor(example["song_id"], dtype=torch.long),
            "genre_id": torch.tensor(example["genre_id"], dtype=torch.long),
            "time_series": torch.tensor(example["time_series"], dtype=torch.float)
        }
