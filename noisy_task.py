# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
import unicodedata
import datetime
import argparse
import codecs
import librosa

from torch.optim.optimizer import Optimizer
from torchaudio.transforms import MFCC, FrequencyMasking, TimeMasking
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import default_collate

try:
    from nipa import nipa_data

    DATASET_PATH = nipa_data.get_data_root('STT_child')
except:
    DATASET_PATH = os.path.join('./data')
    if not os.path.exists(DATASET_PATH):
        share_path = '/datasets/objstrgzip/17_STT_child.zip'
        os.makedirs(DATASET_PATH, exist_ok=True)
        os.system('unzip {} -d {}'.format(share_path, DATASET_PATH))

#
# Dataset Codes
# Grapheme을 데이터로 부터 추출하는 방식 적용 (한글 초 중 종)
#


DICTIONARY = [
    '_', 'ᄀ', 'ᄁ', 'ᄂ', 'ᄃ', 'ᄄ', 'ᄅ', 'ᄆ', 'ᄇ', 'ᄈ', 'ᄉ',
    'ᄊ', 'ᄋ', 'ᄌ', 'ᄍ', 'ᄎ', 'ᄏ', 'ᄐ', 'ᄑ', 'ᄒ', 'ᅡ',
    'ᅢ', 'ᅣ', 'ᅤ', 'ᅥ', 'ᅦ', 'ᅧ', 'ᅨ', 'ᅩ', 'ᅪ', 'ᅫ',
    'ᅬ', 'ᅭ', 'ᅮ', 'ᅯ', 'ᅰ', 'ᅱ', 'ᅲ', 'ᅳ', 'ᅴ', 'ᅵ',
    'ᆨ', 'ᆩ', 'ᆫ', 'ᆭ', 'ᆮ', 'ᆯ', 'ᆰ', 'ᆲ', 'ᆶ', 'ᆷ',
    'ᆸ', 'ᆹ', 'ᆺ', 'ᆻ', 'ᆼ', 'ᆽ', 'ᆾ', 'ᇀ', 'ᇁ', 'ᇂ', '#'
]  # for ctc, _ for padding / 위의 ㄱ 과 아래의 ㄱ 은 초성과 종성의 차이
VOC_SIZE = len(DICTIONARY)


def text2idx(text):
    return [DICTIONARY.index(c) for c in unicodedata.normalize('NFD', text) if c in DICTIONARY[1:-1]] + [VOC_SIZE]


def idx2text(indices):
    if indices[-1] == VOC_SIZE:
        indices = indices[:-1]
    chrs = [DICTIONARY[idx] for idx in indices if idx != VOC_SIZE]
    return unicodedata.normalize('NFC', ''.join(chrs))


class CustomDataset(Dataset):

    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        self.labels = {}

        self.label_path = os.path.join(root, self.phase, self.phase + '_label.txt')
        with codecs.open(self.label_path, 'r', encoding='utf8') as f:
            file_list = []
            label_list = []
            max_length = 0
            for line in f.readlines()[0:]:
                v = line.strip().split()
                file_list.append(v[0])

                if self.phase != 'test':
                    label = v[1:]
                    label_list.append(label)
                else:
                    label_list.append(None)

        ls = [(os.path.join(self.root, self.phase, f), l) for f, l in zip(file_list, label_list)]
        if phase == 'train':
            ls = sorted(ls, key=lambda x: os.path.getsize(x[0]))

        self.labels = ls

    def __getitem__(self, index):
        sound_path, text = self.labels[index]

        if self.phase != 'test':
            text = text[0]
            txt_arr = np.array(text2idx(text))
        else:
            txt_arr = np.array([])

        with open(sound_path, 'rb') as opened_pcm_file:
            buf = opened_pcm_file.read()
            pcm_data = np.frombuffer(buf, dtype='int16').astype(np.float32)
            wav = (pcm_data - np.iinfo(np.int16).min) / (np.iinfo(np.int16).max - np.iinfo(np.int16).min)
            wav = wav * 2 - 1

        # 앞 뒤 공백 제거
        # 50 부터 시작하는 이유는, 마이크 켤 때 발생하는 피크 노이즈를 없애기 위함.
        wav = librosa.effects.trim(wav[50:], top_db=30)[0]

        # make masks
        wav_mask = np.ones_like(wav)
        txt_mask = np.ones_like(txt_arr)

        return wav, txt_arr, wav_mask, txt_mask

    def __len__(self):
        return len(self.labels)

    def get_label_file(self):
        return self.label_path


#
# 길이가 서로 다른 데이터 파일을 유사한 데이터 끼리 뭉쳐주는 역할을 하는 데이터셋 샘플러입니다.
#
class BucketRandomBatchSampler(Sampler):
    """
    Chunking samples into buckets and sample bucket id randomly for each mini batch.
    """

    def __init__(self, data_source: Dataset, n_buckets: int, batch_size: int, skip_last_bucket: bool = False):
        assert len(data_source) > n_buckets * batch_size, 'Data size is too small to use bucket sampler !'
        self.n_buckets = n_buckets
        self.data_size = len(data_source)
        self.batch_size = batch_size
        self.bucket_size = int(math.ceil(self.data_size / self.n_buckets))
        self.bucket_size -= self.bucket_size % batch_size

        if self.n_buckets <= 0:
            raise ValueError("the num of buckets has to be a positive value.")

        self.skip_last_bucket = skip_last_bucket

    def __iter__(self):
        # copy buckets and shuffle indices
        buckets = self.buckets
        for idx in range(len(buckets)):
            np.random.shuffle(buckets[idx])

        # pop up indices
        while buckets:
            bucket_id = np.random.choice(range(len(buckets)))
            ids = buckets[bucket_id][-self.batch_size:]  # pick last
            buckets[bucket_id] = buckets[bucket_id][:-self.batch_size]
            if not buckets[bucket_id]:
                buckets.pop(bucket_id)
            yield ids

    @property
    def buckets(self):
        return [list(range(i * self.bucket_size, (i + 1) * self.bucket_size))
                for i in range(self.n_buckets - int(self.skip_last_bucket))]

    def __len__(self):
        return self.bucket_size * self.n_buckets // self.batch_size


#
# Bucket 샘플러와, batch화 하는 코드를 커스터마이징한 코드입니다.
#
class SpeechDataLoader(DataLoader):

    def __init__(self, dataset, batch_size: int, num_workers: int,
                 n_buckets: int = 10, is_bucket: bool = True, skip_last_bucket: bool = False, is_shuffle: bool = False):

        batch_sampler = None
        if is_bucket:
            batch_sampler = BucketRandomBatchSampler(
                dataset, n_buckets=n_buckets, batch_size=batch_size, skip_last_bucket=skip_last_bucket)
        # call super
        super().__init__(dataset,
                         num_workers=num_workers,
                         collate_fn=self.pad_collate_fn,
                         pin_memory=True,
                         batch_size=(1 if is_bucket else batch_size),
                         shuffle=(not is_bucket and is_shuffle),
                         batch_sampler=batch_sampler)

    @staticmethod
    def pad_collate_fn(batch) -> torch.tensor:
        """
        Matching lengths in using zero-pad
        :param batch: mini batch by sampled on dataset
        :return: collated tensor
        """
        if len(batch) > 1:
            # do zero-padding
            result = []
            for i in range(len(batch[0])):
                # apply padding on dataset
                sub_batch = [x[i] for x in batch]
                # check diff dims
                if not isinstance(sub_batch[0], np.ndarray):
                    # if list of float or int
                    assert all([type(x) == type(sub_batch[0]) for x in sub_batch[1:]])
                    if isinstance(sub_batch[0], int):
                        sub_batch = torch.LongTensor(sub_batch)
                    elif isinstance(batch[0], float):
                        sub_batch = torch.DoubleTensor(sub_batch)

                elif any(list(map(lambda x: x.shape != sub_batch[0].shape, sub_batch[1:]))):
                    sub_batch = torch.from_numpy(__class__.__pad_zero(sub_batch))
                else:
                    sub_batch = torch.from_numpy(np.concatenate(np.expand_dims(sub_batch, axis=0)))
                result.append(sub_batch)
            return result
        else:
            if None in batch:
                return None
            else:
                return default_collate(batch)

    @staticmethod
    def __pad_zero(sub_batch) -> np.ndarray:
        dims = [b.shape for b in sub_batch]

        max_dims = list(dims[0])
        for d_li in dims[1:]:
            for d_idx in range(len(d_li)):
                if max_dims[d_idx] < d_li[d_idx]:
                    max_dims[d_idx] = d_li[d_idx]

        temp = np.zeros((len(sub_batch), *max_dims), dtype=sub_batch[0].dtype)
        for i, b in enumerate(sub_batch):
            if len(b.shape) == 1:
                temp[i, :b.shape[0]] = b
            elif len(b.shape) == 2:
                temp[i, :b.shape[0], :b.shape[1]] = b
            elif len(b.shape) == 3:
                temp[i, :b.shape[0], :b.shape[1], :b.shape[2]] = b
            else:
                raise ValueError
        return temp


def data_loader(root, phase='train', batch_size=16, n_buckets=2):
    if phase == 'train':
        is_bucket = True
    else:
        is_bucket = False
    dataset = CustomDataset(root, phase)
    dataloader = SpeechDataLoader(dataset=dataset, batch_size=batch_size, num_workers=0, is_bucket=is_bucket,
                                  n_buckets=n_buckets)
    return dataloader, dataset.get_label_file()


#
# Optimizer from reference
# https://github.com/clovaai/AdamP
#
class AdamP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        x_norm = x.norm(dim=1).add_(eps)
        y_norm = y.norm(dim=1).add_(eps)
        dot = (x * y).sum(dim=1)

        return dot.abs() / x_norm / y_norm

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(p, grad, perturb, group['delta'], group['wd_ratio'],
                                                         group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.data.add_(-step_size, perturb)

        return loss


#
# Greedy Decoder for inference
#

class GreedyDecoder:
    def __init__(self, labels=DICTIONARY, blank_index=0):
        self.labels = labels + ['<SOS>', '<EOS>']  #
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(self.labels)])
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        for x in range(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
        return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ''
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index] and char not in ['_', '#'] + self.labels[-2:]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                else:
                    string = string + char
        return string

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.
        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs, 2)
        strings = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                          remove_repetitions=True)
        # convert from chrs to normal korean text
        texts = []
        for text in strings:
            s = ''.join([c for c in unicodedata.normalize('NFC', ''.join(text)) if c not in DICTIONARY])
            texts.append(s)
        return texts


#
# Pytorch Model Codes
#
# Attention Module
class Attention(nn.Module):
    """
    Location-based
    """

    def __init__(self, dec_dim, enc_dim, conv_dim, attn_dim, smoothing=False):
        super(Attention, self).__init__()
        self.dec_dim = dec_dim
        self.enc_dim = enc_dim
        self.conv_dim = conv_dim
        self.attn_dim = attn_dim
        self.smoothing = smoothing
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.attn_dim, kernel_size=3, padding=1)

        self.W = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.V = nn.Linear(self.enc_dim, self.attn_dim, bias=False)

        self.fc = nn.Linear(attn_dim, 1, bias=True)
        self.b = nn.Parameter(torch.rand(attn_dim))

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, queries, values, last_attn):
        """
        param:quries: Decoder hidden states, Shape=(B,1,dec_D)
        param:values: Encoder outputs, Shape=(B,enc_T,enc_D)
        param:last_attn: Attention weight of previous step, Shape=(batch, enc_T)
        """
        batch_size = queries.size(0)
        dec_feat_dim = queries.size(2)
        enc_feat_len = values.size(1)

        # conv_attn = (B, enc_T, conv_D)
        conv_attn = torch.transpose(self.conv(last_attn.unsqueeze(dim=1)), 1, 2)

        # (B, enc_T)
        score = self.fc(self.tanh(
            self.W(queries) + self.V(values) + conv_attn + self.b
        )).squeeze(dim=-1)

        if self.mask is not None:
            score.data.masked_fill_(self.mask, -float('inf'))

        # attn_weight : (B, enc_T)
        if self.smoothing:
            score = torch.sigmoid(score)
            attn_weight = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn_weight = self.softmax(score)

            # (B, 1, enc_T) * (B, enc_T, enc_D) -> (B, 1, enc_D)
        context = torch.bmm(attn_weight.unsqueeze(dim=1), values)

        return context, attn_weight


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class RNNP(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, dropout=0.):
        super(RNNP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True, dropout=dropout)
        # self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj = SequenceWise(nn.GroupNorm(1, hidden_size))
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        self.rnn.flatten_parameters()
        x, h = self.rnn(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        x = self.proj(x)
        return x


class ConvLSTMP(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bidirectional=False, rnn_cell='gru',
                 variable_lengths=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.variable_lengths = variable_lengths

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        """
        Copied from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
        Copyright (c) 2017 Sean Naren
        MIT License
        """
        outputs_channel = 32
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, outputs_channel, kernel_size=(21, 11), stride=(2, 2), padding=(10, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(outputs_channel, outputs_channel, kernel_size=(11, 11), stride=(2, 1), padding=(5, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout(0.1)
        ))

        rnn_input_dims = int(math.floor(input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_dims = int(math.floor(rnn_input_dims + 2 * 5 - 11) / 2 + 1)
        rnn_input_dims *= outputs_channel

        rnns = []
        rnn = RNNP(input_size=rnn_input_dims, hidden_size=self.hidden_size, rnn_type=nn.LSTM,
                   bidirectional=self.bidirectional)
        rnns.append(('0', rnn))
        for x in range(n_layers - 1):
            rnn = RNNP(input_size=self.hidden_size, hidden_size=self.hidden_size, rnn_type=nn.LSTM,
                       bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))

        self.fc = SequenceWise(
            nn.Sequential(
                nn.GroupNorm(1, hidden_size),
                nn.Dropout(0.5),
                nn.Linear(self.hidden_size, VOC_SIZE + 1, bias=False)
            )
        )

        self.ctc_loss = nn.CTCLoss(blank=VOC_SIZE - 1)  # vocab_size - 1 : blank vocab_size : eos

    def calculate_ctc_loss(self, ctc_logits: torch.Tensor, chr_target: torch.Tensor, chr_lengths: torch.Tensor):
        T, N, C = ctc_logits.size()
        input_len = torch.LongTensor([T] * N)
        ctc_loss = self.ctc_loss(ctc_logits.log_softmax(2), chr_target, input_len, chr_lengths)
        return ctc_loss

    def forward(self, mfcc, mfcc_length, chr_target=None, chr_lengths=None):
        output_lengths = self.get_seq_lens(mfcc_length)
        x, _ = self.conv(mfcc, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            rnn.flatten_parameters()
            x = rnn(x)

        x = self.fc(x)

        if chr_target is not None and chr_lengths is not None:
            loss = self.calculate_ctc_loss(x, chr_target, chr_lengths)
            return x, loss
        else:
            return x, _

    def get_seq_lens(self, input_length):
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)

        return seq_len.int()


#
# Spectrogram Helper : wav 에 맞추어져 있는 mask 를 spectrogram level로 변형시켜 주는 모듈
#
class SpectrogramMasker(nn.Module):

    def __init__(self, win_length: int = 400, hop_length: int = 200):
        super().__init__()
        self.win_length = win_length
        self.conv = nn.Conv1d(
            1, 1, self.win_length, stride=hop_length, padding=0, bias=False).cuda()
        torch.nn.init.constant_(self.conv.weight, 1. / self.win_length)

    def forward(self, wav_mask: torch.tensor) -> torch.tensor:
        # make mask
        with torch.no_grad():
            wav_mask = F.pad(wav_mask, [0, self.win_length // 2], value=0.)
            wav_mask = F.pad(wav_mask, [self.win_length // 2, 0], value=1.)
            mel_mask = self.conv(wav_mask.float().unsqueeze(1)).squeeze(1)
            mel_mask = torch.ceil(mel_mask)
        return mel_mask


#
# Main Running Code
#
def save_model(model_name, model, optimizer, step, loss, is_best=False, scheduler=None, save_all=False):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if scheduler is not None:
        state.update({'scheduler': scheduler.state_dict()})
    if save_all:
        torch.save(state, os.path.join(model_name + '_{}_{:.6f}.pth'.format(step, loss)))
    if is_best:
        torch.save(state, os.path.join(model_name + '_best.pth'.format(step, loss)))
    print('model saved')


def load_model(file_path, model, optimizer=None, scheduler=None):
    state = torch.load(file_path)
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')


if __name__ == '__main__':
    # setup seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1234)
    np.random.seed(1234)

    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--lr", type=float, default=0.0003)
    args.add_argument("--learning_anneal", type=float, default=1.02)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=180)
    args.add_argument("--print_iter", type=int, default=10)
    args.add_argument("--model_name", type=str, default="model")
    args.add_argument("--pretrained_path", type=str, default="model_best.pth")
    args.add_argument("--prediction_file", type=str, default="prediction.txt")
    args.add_argument("--batch", type=int, default=32)
    args.add_argument("--is_rand_mask", type=bool, default=True)
    args.add_argument("--mode", type=str, default="train")

    config = args.parse_args()

    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    model_name = config.model_name
    pretrained_path = config.pretrained_path
    prediction_file = config.prediction_file
    batch = config.batch
    mode = config.mode
    is_rand_mask = config.is_rand_mask
    learning_anneal = config.learning_anneal

    # create model
    model = ConvLSTMP(
        input_size=80, hidden_size=512, n_layers=5, bidirectional=True, rnn_cell='lstm', variable_lengths=False
    )

    # make criterion
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # mfcc function
    mfcc_func = MFCC(n_mfcc=80, log_mels=True, melkwargs={
        'n_fft': 400, 'hop_length': 200, 'n_mels': 80, 'f_min': 60
    })

    spec_masker = SpectrogramMasker(400, 200)

    if mode == 'test':
        load_model(pretrained_path, model)

    if cuda:
        model = model.cuda()
        mfcc_func = mfcc_func.cuda()
        spec_masker = spec_masker.cuda()

    print(model)

    print("------------------------------------------------------------")
    total_params = sum(p.numel() for p in model.parameters())
    print("num of parameter : ", total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num of trainable_ parameter :", trainable_params)
    print("------------------------------------------------------------", flush=True)

    #
    # Train 코드
    #
    if mode == 'train':
        # set optimizer
        optimizer = AdamP(
            [param for param in model.parameters() if param.requires_grad],
            lr=base_lr, weight_decay=0.1)

        scheduler = None

        # get data loader
        train_dataloader, _ = data_loader(root=DATASET_PATH, phase='train', batch_size=batch, n_buckets=4)
        validate_dataloader, validate_label_file = data_loader(root=DATASET_PATH, phase='validate', batch_size=batch,
                                                               n_buckets=0)
        time_ = datetime.datetime.now()
        num_batches = len(train_dataloader)

        best_valid_loss = 999999.

        # train
        for epoch in range(num_epochs):
            model.train()

            train_loss = 0.

            for iter_, data in enumerate(train_dataloader):
                # fetch train data
                wav, txt_arr, wav_mask, txt_mask = data
                wav, txt_arr, wav_mask, txt_mask = wav.cuda(), txt_arr.cuda(), wav_mask.cuda(), txt_mask.cuda()
                target = txt_arr[:, 1:]

                # wav to mfcc
                mfcc = mfcc_func(wav.unsqueeze(1))
                if is_rand_mask:
                    T = mfcc.size(-1)
                    rand_mask = nn.Sequential(
                        TimeMasking(int(T * 0.3)),
                        FrequencyMasking(20)
                    )
                    mfcc = rand_mask(mfcc)

                # mask mfcc scale
                with torch.no_grad():
                    mfcc_mask = spec_masker(wav_mask)

                # update weight
                logit, loss = model(mfcc, mfcc_mask.sum(-1).long(), chr_target=txt_arr,
                                    chr_lengths=txt_mask.sum(-1).long())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 400)  # max norm 400
                optimizer.step()

                train_loss += loss.item()

                if (iter_ + 1) % print_iter == 0:
                    elapsed = datetime.datetime.now() - time_
                    expected = elapsed * (num_batches / print_iter)
                    _epoch = epoch + ((iter_ + 1) / num_batches)
                    print('[{:.3f}/{:d}] loss({}) '
                          'elapsed {} expected per epoch {}'.format(
                        _epoch, num_epochs, loss.item(), elapsed, expected), flush=True)
                    time_ = datetime.datetime.now()

            # learning rate annealing
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / learning_anneal

            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

            train_loss /= len(train_dataloader)

            # validate
            print('--- Validating ...')
            model.eval()
            valid_loss = 0.

            for iter_, data in enumerate(validate_dataloader):
                # fetch train data
                wav, txt_arr, wav_mask, txt_mask = data
                wav, txt_arr, wav_mask, txt_mask = wav.cuda(), txt_arr.cuda(), wav_mask.cuda(), txt_mask.cuda()
                target = txt_arr[:, 1:]

                # wav to mfcc
                mfcc = mfcc_func(wav.unsqueeze(1))

                # mask mfcc scale
                with torch.no_grad():
                    mfcc_mask = spec_masker(wav_mask)
                    # forward
                    logit, loss = model(mfcc, mfcc_mask.sum(-1).long(), chr_target=txt_arr,
                                        chr_lengths=txt_mask.sum(-1).long())

                    # summation
                    valid_loss += loss.item()

            valid_loss /= len(validate_dataloader)

            if best_valid_loss > valid_loss:
                is_best = True
                best_valid_loss = valid_loss
            else:
                is_best = False

            # save model
            save_model(model_name, model, optimizer, epoch, valid_loss, is_best, scheduler)

            time_ = datetime.datetime.now()
            elapsed = datetime.datetime.now() - time_
            print('[epoch {}] elapsed: {}, train : loss {} / valid : loss {} '.format(
                epoch + 1, elapsed, train_loss, valid_loss), flush=True)

    #
    # predict 코드 내용
    #
    elif mode == 'test':
        model.eval()

        decoder = GreedyDecoder()
        # get data loader
        test_dataloader, _ = data_loader(root=DATASET_PATH, phase='test', batch_size=1, n_buckets=0)

        test_texts = []
        for iter_, data in enumerate(test_dataloader):
            # fetch train data
            wav, txt_arr, wav_mask, txt_mask = data
            wav, txt_arr, wav_mask, txt_mask = wav.cuda(), txt_arr.cuda(), wav_mask.cuda(), txt_mask.cuda()

            # wav to mfcc
            mfcc = mfcc_func(wav.unsqueeze(1))

            # mask mfcc scale
            with torch.no_grad():
                mfcc_mask = spec_masker(wav_mask)

                ctc_logits, _ = model(mfcc, mfcc_mask.sum(-1).long())
                logit = ctc_logits.transpose(0, 1)  # T N C  > N T C

            # decode from logits to text
            texts = decoder.decode(logit)
            test_texts.extend(texts)
            print('Proceed {} / {}'.format(iter_, len(test_dataloader)))

        # 결과 파일 생성
        label_path = os.path.join(DATASET_PATH, 'test', 'test_label.txt')
        with codecs.open(label_path, 'r', encoding='utf8') as f:
            file_list = []
            for line in f.readlines()[0:]:
                v = line.strip().split()
                file_list.append(v[0])

        with open(prediction_file, 'w') as w:
            for file_path, pred_text in zip(file_list, test_texts):
                w.write('{} {}\n'.format(file_path, pred_text))
        print('finish!')