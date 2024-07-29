import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TorchMusicDataset(Dataset):
    def __init__(self, dataset_folder, seconds_per_example):
        self.dataset_folder = dataset_folder
        self.seconds_per_example = seconds_per_example
        self.examples = [
            os.path.join(self.dataset_folder, example_file)
            for example_file in os.listdir(self.dataset_folder)
        ]
        self.sampling_rate = 44100

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = np.load(self.examples[index], allow_pickle=True).tolist()
        time_series = np.array(example["time_series"])
        # Cut to complete seconds
        n_steps_int_seconds = self.sampling_rate * self.seconds_per_example
        # Reshape to one second per feature
        time_series = time_series[:n_steps_int_seconds].reshape((self.sampling_rate, self.seconds_per_example))
        label = (
            np.ones((self.sampling_rate, 1)) if example["genre_id"] == 1
            else np.zeros((self.sampling_rate, 1))
        )
        time_series = np.concatenate((time_series, label), axis=1)
        return torch.tensor(time_series, dtype=torch.float32)
