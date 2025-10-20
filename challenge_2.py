from pathlib import Path
import math
import os
import random
from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.functional import l1_loss
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset
from braindecode.models import EEGNeX
from eegdash import EEGChallengeDataset


class DatasetWrapper(BaseDataset):
    def __init__(
        self,
        dataset: EEGWindowsDataset,
        crop_size_samples: int,
        target_name: str = "externalizing",
        seed=None,
    ):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name = target_name
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]

        target = self.dataset.description[self.target_name]
        target = float(target)

        # Additional information:
        infos = {
            "subject": self.dataset.description["subject"],
            "sex": self.dataset.description["sex"],
            "age": float(self.dataset.description["age"]),
            "task": self.dataset.description["task"],
            "session": self.dataset.description.get("session", None) or "",
            "run": self.dataset.description.get("run", None) or "",
        }

        # Randomly crop the signal to the desired length:
        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples, f"{i_stop=} {i_start=}"
        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        i_start = i_start + start_offset
        i_stop = i_start + self.crop_size_samples
        X = X[:, start_offset : start_offset + self.crop_size_samples]

        return X, target, (i_window_in_trial, i_start, i_stop), infos

# --- Main execution block ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        msg = "CUDA-enabled GPU found. Training should be faster."
    else:
        msg = (
            "No GPU found. Training will be carried out on CPU, which might be "
            "slower.\n\nIf running on Google Colab, you can request a GPU runtime by"
            " clicking\n`Runtime/Change runtime type` in the top bar menu, then "
            "selecting 'T4 GPU'\nunder 'Hardware accelerator'."
        )
    print(msg)

    # The first step is to define the cache folder!
    DATA_DIR = Path("./data")

    # Creating the path if it does not exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # We define the list of releases to load.
    release_list = ["R5"]

    all_datasets_list = [
        EEGChallengeDataset(
            release=release,
            task="contrastChangeDetection",
            mini=True,
            description_fields=[
                "subject",
                "session",
                "run",
                "task",
                "age",
                "gender",
                "sex",
                "p_factor",
            ],
            cache_dir=DATA_DIR,
        )
        for release in release_list
    ]
    print("Datasets loaded")
    sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
              "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]

    all_datasets = BaseConcatDataset(all_datasets_list)
    print(all_datasets.description)

    raws = Parallel(n_jobs=os.cpu_count())(
        delayed(lambda d: d.raw)(d) for d in all_datasets.datasets
    )

    SFREQ = 100

    # Filter out recordings that are too short
    all_datasets = BaseConcatDataset(
        [
            ds
            for ds in all_datasets.datasets
            if not ds.description.subject in sub_rm
            and ds.raw.n_times >= 4 * SFREQ
            and len(ds.raw.ch_names) == 129
            and not math.isnan(ds.description["p_factor"])
        ]
    )

    # Create 4-seconds windows with 2-seconds stride
    windows_ds = create_fixed_length_windows(
        all_datasets,
        window_size_samples=4 * SFREQ,
        window_stride_samples=2 * SFREQ,
        drop_last_window=True,
    )

    # Wrap each sub-dataset in the windows_ds
    windows_ds = BaseConcatDataset(
        [DatasetWrapper(ds, crop_size_samples=2 * SFREQ) for ds in windows_ds.datasets]
    )

    # Initialize model
    model = EEGNeX(n_chans=129, n_outputs=1, n_times=2 * SFREQ).to(device)

    # Specify optimizer
    optimizer = optim.Adamax(params=model.parameters(), lr=0.002)

    print(model)

    # Create PyTorch Dataloader
    # Set num_workers to 0 on Windows if you still have issues, or 1 if it works.
    num_workers = 0 if os.name == 'nt' else os.cpu_count()
    dataloader = DataLoader(windows_ds, batch_size=128, shuffle=True, num_workers=num_workers)

    n_epochs = 1

    # Train model for 1 epoch
    for epoch in range(n_epochs):

        for idx, batch in enumerate(dataloader):
            # Reset gradients
            optimizer.zero_grad()

            # Unpack the batch
            X, y, crop_inds, infos = batch
            X = X.to(dtype=torch.float32, device=device)
            y = y.to(dtype=torch.float32, device=device).unsqueeze(1)

            # Forward pass
            y_pred = model(X)

            # Compute loss
            loss = l1_loss(y_pred, y)
            print(f"Epoch {0} - step {idx}, loss: {loss.item()}")

            # Gradient backpropagation
            loss.backward()
            optimizer.step()

    # Finally, we can save the model for later use
    torch.save(model.state_dict(), "weights_challenge_2.pt")
    print("Model saved as 'weights_challenge_2.pt'")


if __name__ == '__main__':
    main()