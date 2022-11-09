import os
from fairseq import utils, search
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data import data_utils, LanguagePairDataset, ConcatDataset, IndexedRawTextDataset
from fairseq.data.dictionary import Dictionary
import fairseq.data.indexed_dataset as indexed_dataset
from fairseq.sequence_generator import SequenceGenerator
from fairseq.tokenizer import tokenize_line
import itertools
import torch
from torch import Tensor
import numpy as np
import logging
from typing import Dict, List, Optional
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
        self.eos_index = self.bos_index
        self.pad_index = self.add_symbol(pad)
        self.nspecial = len(self.symbols)
        self.unk_index = 99999

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

    @property
    def label_dictionary(self):
        return self.label_dict

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

    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"
        search_strategy = search.BeamSearch(self.target_dictionary)
        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        return SequenceGeneratorWithDD(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens, constraints=constraints)


class SequenceGeneratorWithDD(SequenceGenerator):
    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate( #初始化incremental_states为[{}]，用来指示增量解码状态
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if 'src_tokens' in net_input:
            src_tokens = net_input['src_tokens']
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1) #src_lengths是实际样本长度
        elif 'source' in net_input:
            src_tokens = net_input['source']
            src_lengths = (
                net_input['padding_mask'].size(-1) - net_input['padding_mask'].sum(-1)
                if net_input['padding_mask'] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception('expected src_tokens or source in net input')

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2] #src_len是pad后组装成batch的样本的长度，src_lengths是实际样本长度
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError("Target-side constraints were provided, but search method doesn't support them")

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min( #允许的解码最大长度
                int(self.max_len_a * src_len + self.max_len_b),# 命令行参数设置
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,# 模型最大能力，包括transformer和位置编码的最大长度
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam

        encoder_outs = self.model.forward_encoder(net_input)
        dd_labels_outs = [
                model.enc_classification_layer(encoder_outs[0].encoder_out)
                for model in self.model.models
            ]
        dd_labels_outs = torch.argmax(dd_labels_outs[0].transpose(0,1), dim=-1)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1) #(bsz*beam_size)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order) #将编码器输出维度bsz——>bsz*beam_size，用于解码beam search
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = ( #torch.Size([B*beam, 201])
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = ( #torch.Size([B*beam, 202])
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad) #初始化全为pad
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token #上一个step解码器的输出，也就是本step解码器的输入，第一个token是eos
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = ( #torch.Size([B, beam_size])，初始化全为False
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate( #list，长度B
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of information about the hypothesis being finalized at each step

        finished = [ #list，长度B，初始化全为False
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens) #torch.Size([B, 1])
        cand_offsets = torch.arange(0, cand_size).type_as(tokens) #torch.Size([2 * beam_size])

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            # print(f'step: {step}')
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            lprobs, avg_attn_scores = self.model.forward_decoder( #输入到解码器，第一步step=0，只输入tokens中的第一个元素eos
                tokens[:, : step + 1],
                encoder_outs,
                incremental_states,
                self.temperature,
            ) #lprobs torch.Size([2480, 42720])，表示这一步输出的概率密度，avg_attn_scores torch.Size([B*beam, srclen])
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len: #只输出eos，强制结束
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf #强制不输出eos

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty( #torch.Size([B * beam_size, T, 202])，attn存储的是所有step的attention值
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)，beam_search
            cand_scores, cand_indices, cand_beams = self.search.step( #cand_beams是属于第几个beam
                step,
                lprobs.view(bsz, -1, self.vocab_size), #torch.Size([496, 5, 42720])
                scores.view(bsz, beam_size, -1)[:, :, :step], #torch.Size([496, 5, 0])
            )
            # cand_scores torch.Size([496, 10])，cand_indices torch.Size([496, 10])，cand_beams torch.Size([496, 10])
            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets) #torch.Size([496, 10])

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf) #torch.Size([496, 10])
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents) #num_remaining_sent是batch中还未结束的句子数，只要不是所有beam都eos，该句就没有结束

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0: #batch中的所有句子都结束了
                break
            assert step < max_len

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(bsz, dtype=torch.bool, device=cand_indices.device)
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(bsz, device=cand_indices.device).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk( #从小往大
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1) #active_bbsz_idx和active_scores组成了下一个step需要的scores
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx #去掉在该stepeos的beam，目前的tokens还没有加上该step输出的token
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather( #把该step输出的token加到tokens的最后
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor([float(elem["score"].item()) for elem in finalized[sent]])
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices] #根据scores从高到低对finalized重新排序
            finalized[sent] = torch.jit.annotate(List[Dict[str, Tensor]], finalized[sent])

        return finalized, dd_labels_outs
