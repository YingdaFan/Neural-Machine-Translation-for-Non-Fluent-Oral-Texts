import os
from fairseq import utils, metrics
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data import data_utils, LanguagePairDataset, ConcatDataset, IndexedRawTextDataset
from fairseq.data.dictionary import Dictionary
import fairseq.data.indexed_dataset as indexed_dataset
from fairseq.tokenizer import tokenize_line
import itertools
import torch
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

def collate_triple(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    id = torch.LongTensor([s['id'] for s in samples]) # 一个batch的id
    src_tokens = merge(
        'source', left_pad=left_pad_source,
        pad_to_length=pad_to_length['source'] if pad_to_length is not None else None
    )
    # sort by descending source length
    src_lengths = torch.LongTensor([
        s['source'].ne(pad_idx).long().sum() for s in samples
    ])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge(
            'target', left_pad=left_pad_target,
            pad_to_length=pad_to_length['target'] if pad_to_length is not None else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([
            s['target'].ne(pad_idx).long().sum() for s in samples
        ]).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get('prev_output_tokens', None) is not None:
            prev_output_tokens = merge('prev_output_tokens', left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length['target'] if pad_to_length is not None else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    label = merge('label', left_pad=left_pad_source)
    label = label.index_select(0, sort_order)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'label': label
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens.index_select(0, sort_order)

    return batch


class LanguageTripleDataset(LanguagePairDataset):
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        label=None, label_sizes=None, label_dict=None
    ):
        super(LanguageTripleDataset, self).__init__(src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict)
        if label is not None:
            assert len(src) == len(label), "Source and label must contain the same number of examples"
        self.label = label
        self.label_sizes = np.array(label_sizes) if label_sizes is not None else None
        assert (self.label_sizes == self.src_sizes).all() if label_sizes is not None else None, "Source and label must have the same size"
        self.label_dict = label_dict

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        label_item = self.label[index] if self.label is not None else None

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'label': label_item
        }

        return example

    def collater(self, samples, pad_to_length=None):# return a batch of data
        res = collate_triple(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
        )
        return res


class LabelDataset(IndexedRawTextDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_zero=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_zero = append_zero
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.lines.append(line.strip('\n'))
                labels = dictionary.encode_line(line, append_zero=self.append_zero).long()
                self.tokens_list.append(labels)
                self.sizes.append(len(labels))
        self.sizes = np.array(self.sizes)


def make_label_dataset(path, impl, dictionary=None):
    if impl == 'raw' and LabelDataset.exists(path):
        assert dictionary is not None
        return LabelDataset(path, dictionary)
    else:
        return None


def load_label_dataset(path, dictionary=None, dataset_impl=None):
    """A helper function for loading indexed datasets.

    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train')
        dictionary (~fairseq.data.Dictionary): data dictionary
        dataset_impl (str, optional): which dataset implementation to use. If
            not provided, it will be inferred automatically. For legacy indexed
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    if dataset_impl is None:
        dataset_impl = indexed_dataset.infer_dataset_impl(path)
    dataset = make_label_dataset(path, impl=dataset_impl, dictionary=dictionary)
    logger.info('loaded {} examples from: {}'.format(len(dataset), path))
    return dataset


class Dictionary_SequenceLabel(Dictionary):
    def __init__(self, bos="<s>", pad="<pad>"):
        self.pad_word = pad
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.nspecial = len(self.symbols)

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        append_zero=True
    ):
        labels = line_tokenizer(line)
        nlabels = len(labels)
        ids = torch.IntTensor(nlabels + 1 if append_zero else nlabels)
        for i, label in enumerate(labels):
            idx = self.index(label)
            ids[i] = idx
        if append_zero:
            ids[nlabels] = 0
        return ids


@register_task('multitask_dd_translation_task')
class MultitaskDDTranslationTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict, label_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.label_dict = label_dict

    def load_dataset(self, split, **kwargs):
        prefix_translation = os.path.join(self.args.data, '{}.{}-{}.'.format(split, self.args.source_lang, self.args.target_lang))
        prefix_sequencelabel = os.path.join(self.args.data, '{}.{}-{}.'.format(split, self.args.source_lang, "label"))

        src_dataset = data_utils.load_indexed_dataset(prefix_translation + self.args.source_lang, self.src_dict)
        tgt_dataset = data_utils.load_indexed_dataset(prefix_translation + self.args.target_lang, self.tgt_dict)
        label_dataset = load_label_dataset(prefix_sequencelabel + "label", self.label_dict)

        self.datasets[split] = LanguageTripleDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            label_dataset, label_dataset.sizes, self.label_dict
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        label_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format("label")))
        return cls(args, src_dict, tgt_dict, label_dict)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if filename[-9:-4] == "label":
            return Dictionary_SequenceLabel.load(filename)
        else:
            return Dictionary.load(filename)

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        loss_translation = sum(log.get('loss_trans', 0) for log in logging_outputs)
        loss_label = sum(log.get('loss_label', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        metrics.log_scalar('loss_translation', loss_translation / ntokens / math.log(2), round=3)
        metrics.log_scalar('loss_label', loss_label / ntokens / math.log(2), round=3)