# Pytorch-Snippets
This repository will store useful and repetative code snippets for PyTorch.

# Dataset Splitting
(split a Dataset object into 3 sub-datasets: train, val and test. adapted from [here](https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/day02-PyTORCH-and-PyCUDA/PyTorch/21-PyTorch-CIFAR-10-Custom-data-loader-from-scratch.ipynb))
```
class FullTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        assert len(full_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(FullTrainingDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i + self.offset]


def trainTestSplit(dataset, val_share=0.1, test_share=0.15):
    val_offset = int(len(dataset) * (1 - val_share - test_share))
    test_offset = int(len(dataset) * (1 - test_share))
    train_len = val_offset
    test_len = len(dataset) - test_offset
    val_len = len(dataset) - val_offset - test_len
    assert train_len + test_len + val_len == len(dataset)
    return FullTrainingDataset(dataset, 0, val_offset), \
           FullTrainingDataset(dataset, val_offset, val_len), \
           FullTrainingDataset(dataset, test_offset, test_len)
```
