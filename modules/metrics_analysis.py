import inspect
import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import Union
from copy import deepcopy
import re

import numpy as np
import torch

from fastNLP import MetricBase, ClassifyFPreRecMetric, SpanFPreRecMetric
from .utils import seq_len_to_mask
from fastNLP import Vocabulary


class ClassifyFPreRecMetricAnalysis(ClassifyFPreRecMetric):
    
    def __init__(self, tag_vocab=None, pred=None, target=None, seq_len=None, ignore_labels=None,
                 only_gross=True, f_type='micro', beta=1):
        
        ClassifyFPreRecMetric.__init__(self, tag_vocab, pred, target, seq_len, ignore_labels,
                 only_gross, f_type, beta)
        self.error = {}

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if seq_len is not None and not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
        else:
            masks = torch.ones_like(target).long().to(target.device)

        masks = masks.eq(1)

        if pred.dim() == target.dim():
            if torch.numel(pred) !=torch.numel(target):
                raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have same dimensions with target, they should have same element numbers. while target have "
                               f"element numbers:{torch.numel(target)}, pred have element numbers: {torch.numel(pred)}")

            pass
        elif pred.dim() == target.dim() + 1:
            pred = pred.argmax(dim=-1)
            if seq_len is None and target.dim() > 1:
                warnings.warn("You are not passing `seq_len` to exclude pad when calculate accuracy.")
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        target = target.to(pred)
        target = target.masked_select(masks)
        pred = pred.masked_select(masks)

        self.error = {}
        if ((pred != target).sum()) != 0:
            self.error = {'pred':pred, 'tgt':target}

        target_idxes = set(target.reshape(-1).tolist())
        for target_idx in target_idxes:
            self._tp[target_idx] += torch.sum((pred == target_idx).long().masked_fill(target != target_idx, 0)).item()
            self._fp[target_idx] += torch.sum((pred == target_idx).long().masked_fill(target == target_idx, 0)).item()
            self._fn[target_idx] += torch.sum((pred != target_idx).long().masked_fill(target != target_idx, 0)).item()

    def get_error(self):
        return self.error


    def get_metric(self, reset=True):
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._fn.keys())
            tags.update(set(self._fp.keys()))
            tags.update(set(self._tp.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                if self.tag_vocab is not None:
                    tag_name = self.tag_vocab.to_word(tag)
                else:
                    tag_name = int(tag)
                tp = self._tp[tag]
                fn = self._fn[tag]
                fp = self._fp[tag]
                f, pre, rec = _compute_f_pre_rec(self.beta_square, tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''防止无tag的情况
                    f_key = 'f-{}'.format(tag_name)
                    pre_key = 'pre-{}'.format(tag_name)
                    rec_key = 'rec-{}'.format(tag_name)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec = _compute_f_pre_rec(self.beta_square,
                                             sum(self._tp.values()),
                                             sum(self._fn.values()),
                                             sum(self._fp.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        if reset:
            self._tp = defaultdict(int)
            self._fp = defaultdict(int)
            self._fn = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result






class SpanFPreRecMetricAnalysis(SpanFPreRecMetric):

    def __init__(self, tag_vocab, pred=None, target=None, seq_len=None, encoding_type=None, ignore_labels=None,
                 only_gross=True, f_type='micro', beta=1):
        
        SpanFPreRecMetric.__init__(self, tag_vocab, pred, targete, seq_len, encoding_type, ignore_labels,
                 only_gross, f_type, beta)

    def evaluate(self, pred, target, seq_len):
        
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if pred.size() == target.size() and len(target.size()) == 2:
            pass
        elif len(pred.size()) == len(target.size()) + 1 and len(target.size()) == 2:
            num_classes = pred.size(-1)
            pred = pred.argmax(dim=-1)
            if (target >= num_classes).any():
                raise ValueError("A gold label passed to SpanBasedF1Metric contains an "
                                 "id >= {}, the number of classes.".format(num_classes))
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        batch_size = pred.size(0)
        pred = pred.tolist()
        target = target.tolist()
        for i in range(batch_size):
            pred_tags = pred[i][:int(seq_len[i])]
            gold_tags = target[i][:int(seq_len[i])]

            pred_str_tags = [self.tag_vocab.to_word(tag) for tag in pred_tags]
            gold_str_tags = [self.tag_vocab.to_word(tag) for tag in gold_tags]

            pred_spans = self.tag_to_span_func(pred_str_tags, ignore_labels=self.ignore_labels)
            gold_spans = self.tag_to_span_func(gold_str_tags, ignore_labels=self.ignore_labels)

            for span in pred_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def get_metric(self, reset=True):
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._false_negatives.keys())
            tags.update(set(self._false_positives.keys()))
            tags.update(set(self._true_positives.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = self._true_positives[tag]
                fn = self._false_negatives[tag]
                fp = self._false_positives[tag]
                f, pre, rec = _compute_f_pre_rec(self.beta_square, tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''防止无tag的情况
                    f_key = 'f-{}'.format(tag)
                    pre_key = 'pre-{}'.format(tag)
                    rec_key = 'rec-{}'.format(tag)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec = _compute_f_pre_rec(self.beta_square,
                                             sum(self._true_positives.values()),
                                             sum(self._false_negatives.values()),
                                             sum(self._false_positives.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        if reset:
            self._true_positives = defaultdict(int)
            self._false_positives = defaultdict(int)
            self._false_negatives = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result


def _compute_f_pre_rec(beta_square, tp, fn, fp):
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * pre * rec / (beta_square * pre + rec + 1e-13)

    return f, pre, rec


# def _prepare_metrics(metrics):
#     _metrics = []
#     if metrics:
#         if isinstance(metrics, list):
#             for metric in metrics:
#                 if isinstance(metric, type):
#                     metric = metric()
#                 if isinstance(metric, MetricBase):
#                     metric_name = metric.__class__.__name__
#                     if not callable(metric.evaluate):
#                         raise TypeError(f"{metric_name}.evaluate must be callable, got {type(metric.evaluate)}.")
#                     if not callable(metric.get_metric):
#                         raise TypeError(f"{metric_name}.get_metric must be callable, got {type(metric.get_metric)}.")
#                     _metrics.append(metric)
#                 else:
#                     raise TypeError(
#                         f"The type of metric in metrics must be `fastNLP.MetricBase`, not `{type(metric)}`.")
#         elif isinstance(metrics, MetricBase):
#             _metrics = [metrics]
#         else:
#             raise TypeError(f"The type of metrics should be `list[fastNLP.MetricBase]` or `fastNLP.MetricBase`, "
#                             f"got {type(metrics)}.")
#     return _metrics
