from torch.utils.data import Dataset

from compressai.registry import register_dataset


@register_dataset("EmptyDataset")
class EmptyDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")
