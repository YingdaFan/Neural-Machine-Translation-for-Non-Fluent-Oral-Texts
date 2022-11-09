from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss

@register_criterion('multitask_dd_translation_criterion')
class MultitaskDDTranslationCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, lambda_dd):
        super().__init__(task, sentence_avg, label_smoothing)
        self.lambda_dd = lambda_dd
        label_dict = task.label_dictionary
        self.label_padding_idx = label_dict.pad()

    @staticmethod
    def add_args(parser):
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--lambda_dd', type=float, metavar='L',
                            help='total loss = lambda_dd*loss_dd + (1-labda_dd)*loss_mt')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, loss_trans, loss_label = self.compute_sum_of_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'loss_trans': loss_trans,
            'loss_label': loss_label,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_sum_of_loss(self, model, net_output, sample, reduce=True):
        lprobs_trans = model.get_normalized_probs(net_output[1], log_probs=True)
        lprobs_label = model.get_normalized_probs(net_output[0], log_probs=True)
        lprobs_trans = lprobs_trans.view(-1, lprobs_trans.size(-1))
        lprobs_label = lprobs_label.view(-1, lprobs_label.size(-1))
        target = model.get_targets(sample, net_output[1]).view(-1, 1)
        label = model.get_labels(sample, net_output[0]).view(-1, 1)
        loss_trans, nll_loss_trans = label_smoothed_nll_loss(
            lprobs_trans, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        loss_label, nll_loss_label = label_smoothed_nll_loss(
            lprobs_label, label, self.eps, ignore_index=self.label_padding_idx, reduce=reduce,
        )
        loss_all = self.lambda_dd * loss_label + (1-self.lambda_dd) * loss_trans
        nll_loss_all = self.lambda_dd * nll_loss_label + (1-self.lambda_dd) * nll_loss_trans
        return loss_all, nll_loss_all, loss_trans, loss_label