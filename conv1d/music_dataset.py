import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TorchMusicDataset(Dataset):
    def __init__(self, dataset_folder, seconds_per_example, downsample_ratio=None):
        self.dataset_folder = dataset_folder
        self.seconds_per_example = seconds_per_example
        self.downsample_ratio = downsample_ratio
        self.examples = [
            os.path.join(self.dataset_folder, example_file)
            for example_file in os.listdir(self.dataset_folder)
        ][:100]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = np.load(self.examples[index], allow_pickle=True).tolist()
        time_series = np.array(example["time_series"])
        # Cut to complete seconds
        n_steps_int_seconds = 44100 * self.seconds_per_example
        # Reshape to one second per feature
        time_series = time_series[:n_steps_int_seconds]
        if self.downsample_ratio is not None:
            time_series = time_series[::self.downsample_ratio]
        time_series = time_series.reshape(-1, 1)
        return torch.tensor(time_series, dtype=torch.float32).transpose(0,1)
