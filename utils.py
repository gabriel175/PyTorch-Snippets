import numbers
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as var
from torch import FloatTensor as ft
from torch.utils.data import DataLoader
import torch.autograd as autograd
import os
import shutil
import numpy as np


### DATASETS ###


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


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    return torch.cat([vec, vec.new(pad - vec.size(dim), *vec.size()[1:]).zero_()], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = map(lambda (x, y):
                    (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
        # stack all
        xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        ys = torch.LongTensor(map(lambda x: x[1], batch))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


### GENERAL TRANSFORMATIONS ###


class ListToNumpy(object):
    """ Transform a list to a numpy array """

    def __call__(self, list):
        return np.array(list)


class NumpyToTensor(object):
    """ Transform a numpy array to a Tensor """

    def __init__(self, is_float=True):
        self.is_float = is_float

    def __call__(self, numpy_arr):
        x = torch.from_numpy(numpy_arr)
        if self.is_float:
            return x.float()
        return x


class DimZeroPad(object):
    """ Zero pad a tensor along a given dimension """

    def __init__(self, padding, dim=0):
        assert isinstance(padding, numbers.Number)
        self.padding = padding
        self.dim = dim

    def __call__(self, vec):
        assert (vec.dim() - 1) >= self.dim, "Padded dim doesn't exist!"
        return pad_tensor(vec, self.padding, self.dim)
    
    
### AUDIO TRANSFORMATIONS ###
    
    
class PathToWav(object):
    """ Transform a path to a wav file into a 1d numpy array """

    def __call__(self, path):
        sr, sig = read(path)
        return sig, sr


class ProbabilityTransform(object):
    """ apply transform with a probability """

    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return self.transform(x)
        return x

    def transform(self, x):
        raise NotImplementedError


class RandomWhiteNoise(ProbabilityTransform):
    """ Adds random white noise to signal """

    def __init__(self, noise_strength=0.005, p=0.0):
        self.noise_strength = noise_strength
        super(RandomWhiteNoise, self).__init__(p)


    def transform(self, (sig, sr)):
        wn = np.random.randn(len(sig))
        sig_wn = sig + self.noise_strength * wn
        return sig_wn, sr


class RandomShift(ProbabilityTransform):
    """ Adds a random shift to the signal """

    def __init__(self, shift_percent=0.1, p=1.0):
        self.shift_percent = shift_percent
        super(RandomShift, self).__init__(p)

    def transform(self, (sig, sr)):
        direction = random.choice([-1, 1])
        shift = int(direction * len(sig) * self.shift_percent)
        return np.roll(sig, shift), sr


class WavToSpectrogram(object):
    """ Transform a 1d numpy array to a 2d spectrogram """

    def __call__(self, (sig, sr)):
        frequencies, times, spectrogram = signal.spectrogram(sig, sr)
        return np.transpose(spectrogram)


def save_checkpoint(state, is_best, is_error=False, filename='models/checkpoint'):
    filename += "ok" if not is_error else "fail"
    filename += ".pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'models/model_best.pth.tar')
        with open('models/model_best_summary.txt', 'wb') as f:
            f.write(state['summary'])
    print('==> saved {}'.format('(*)' if is_best else ''))
    
    
### MODEL SAVING/LOADING ####


def load_checkpoint(model, optimizer, path):
    if not path:
        print("==> creating a new model")
        return 0, float("inf")
    if path == '-1':
        print("==> loading best model")
        path = 'models/model_best.pth.tar'
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("==> loaded checkpoint '{}' (epoch {})".format(
            path, checkpoint['epoch']))
        return start_epoch, best_loss
    else:
        print("==> no checkpoint found at '{}'".format(path))
        
        
### RNN ###


def sort_and_pack(tensor, lengths):
    seq_lengths = lengths
    sorted_len, sorted_idx = seq_lengths.sort(0, descending=True)
    index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(tensor)
    sorted_inputs = tensor.gather(0, index_sorted_idx.long())
    packed_seq = torch.nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
    return packed_seq, sorted_idx


def unpack_and_unsort(packed, sorted_idx):
    unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
    # unsort the output
    _, original_idx = sorted_idx.sort(0, descending=False)
    unsorted_idx = original_idx.view(-1, 1, 1).expand_as(unpacked)
    output = unpacked.gather(0, unsorted_idx.long())
    return output


### MODULES ###


class ConvBlock(nn.Module):
    """
    a block of conv + bn + activation
    """

    def __init__(self, in_channel, out_channel, kernel=2, stride=1, padding=0, conv_type='2d', activation='relu', batch_norm=True):
        """
        in_channel, out_channel, kernel=2, stride=1, padding=0 - the same parameters as in pytorch conv1d/conv2d
        conv_type - '1d' / '2d', determines the type of convolution
        activation - the type of activation to use after the convolution
        batch_norm - flag that determines if batch-norm should be used
        """
        super(ConvBlock, self).__init__()
        if activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'hard_tanh':
            activation = nn.Hardtanh(0, 20, inplace=True)
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
        else:
            activation = nn.ReLU()

        if conv_type == '2d':
            self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
                                      nn.BatchNorm2d(out_channel),
                                      activation)
        else:
            self.conv = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel, stride, padding),
                                      nn.BatchNorm1d(out_channel),
                                      activation)

    def forward(self, x):
        return self.conv(x)
    
    
class CollapseConvSeq(nn.Module):
    """
    gets an output of a convolution (batch_size x out_channel x seq_len x feat)
    and outputs it to a (batch_size x seq_len x (feat * out_channel))
    
    it preserves the length of the sequences, just increases the amount of features
    """

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        return x.transpose(1, 2).contiguous()


class Flatten(nn.Module):
    """
    flatten to 1d vector
    """

    def forward(self, x):
        return x.view(x.size(0), -1)