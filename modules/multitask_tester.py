from fastNLP import Tester
from fastNLP import BatchIter, DataSetIter, DataSet
from fastNLP import Sampler, SequentialSampler
from .utils import _CheckError
from .utils import _build_args
from .utils import _check_loss_evaluate
from .utils import _move_dict_value_to_device
from .utils import _get_func_signature
from .utils import _get_model_device
from .utils import _move_model_to_device
from .utils import seq_len_to_mask
from ._parallel_utils import _data_parallel_wrapper
from ._parallel_utils import _model_contains_inner_module
#from .metrics_analysis import _prepare_metrics
from functools import partial

import time

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except:
    from .utils import _pseudo_tqdm as tqdm


class Multitask_Tester(Tester):
    def __init__(self, data, model, metrics=None, batch_size=16, num_workers=0, device=None, verbose=1, use_tqdm=True,
                 **kwargs):
        Tester.__init__(self, data, model, metrics, batch_size, num_workers, device, verbose, use_tqdm,
                 **kwargs)
            

    def test(self):
        self._model_device = _get_model_device(self._model)
        network = self._model
        self._mode(network, is_test=True)
        data_iterator = self.data_iterator
        eval_results = {}
        try:
            with torch.no_grad():
                if not self.use_tqdm:
                    from .utils import _pseudo_tqdm as inner_tqdm
                else:
                    inner_tqdm = tqdm
                with inner_tqdm(total=len(data_iterator), leave=False, dynamic_ncols=True) as pbar:
                    pbar.set_description_str(desc="Test")

                    start_time = time.time()

                    for batch_x, batch_y in data_iterator:
                        _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)
                        pred_dict = self._data_forward(self._predict_func, batch_x)
                        if not isinstance(pred_dict, dict):
                            raise TypeError(f"The return value of {_get_func_signature(self._predict_func)} "
                                            f"must be `dict`, got {type(pred_dict)}.")
                        

                        c_pred = pred_dict['c_pred']
                        for metric in self.metrics:
                            metric({'pred': c_pred}, {'target': batch_y['c'], 'seq_len': batch_y['seq_len']})

                        if self.use_tqdm:
                            pbar.update()

                        for metric in self.metrics:
                            eval_result = metric.get_metric()
                            if not isinstance(eval_result, dict):
                                raise TypeError(f"The return value of {_get_func_signature(metric.get_metric)} must be "
                                                f"`dict`, got {type(eval_result)}")
                            metric_name = metric.get_metric_name()
                            eval_results[metric_name] = eval_result

                    pbar.close()
                    end_time = time.time()
                    test_str = f'Evaluate data in {round(end_time - start_time, 2)} seconds!'
                    if self.verbose >= 0:
                        self.logger.info(test_str)
        except _CheckError as e:
            prev_func_signature = _get_func_signature(self._predict_func)
            _check_loss_evaluate(prev_func_signature=prev_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, pred_dict=pred_dict, target_dict=batch_y,
                                 dataset=self.data, check_level=0)
        finally:
            self._mode(network, is_test=False)
        if self.verbose >= 1:
            logger.info("[tester] \n{}".format(self._format_eval_results(eval_results)))
        return eval_results

    def _data_forward(self, func, x):
        r"""A forward pass of the model. """
        x = _build_args(func, **x)
        y = self._predict_func_wrapper(**x)
        return y
    
    

    def __init__(self, data, model, metrics=None, batch_size=16, num_workers=0, device=None, verbose=1, use_tqdm=True,
                 **kwargs):
        Tester.__init__(self, data, model, metrics, batch_size, num_workers, device, verbose, use_tqdm,
                 **kwargs)
        self.has_c = True if data.has_field('c') else False
        self.errors = {'words':[], 'pred':[], 'tgt':[]}
            

    def test(self):
        self._model_device = _get_model_device(self._model)
        network = self._model
        self._mode(network, is_test=True)
        data_iterator = self.data_iterator
        eval_results = {}
        try:
            with torch.no_grad():
                if not self.use_tqdm:
                    from .utils import _pseudo_tqdm as inner_tqdm
                else:
                    inner_tqdm = tqdm
                with inner_tqdm(total=len(data_iterator), leave=False, dynamic_ncols=True) as pbar:
                    pbar.set_description_str(desc="Test")

                    start_time = time.time()

                    for batch_x, batch_y in data_iterator:
                        _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)
                        pred_dict = self._data_forward(self._predict_func, batch_x)
                        if not isinstance(pred_dict, dict):
                            raise TypeError(f"The return value of {_get_func_signature(self._predict_func)} "
                                            f"must be `dict`, got {type(pred_dict)}.")
                        p_metrics = self.metrics[0]
                        error = p_metrics.get_error()
                        if len(error.keys()) > 0:
                            words = batch_x['words']
                            max_len = target.size(1)
                            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
                            words = words.masked_select(masks)
                            pred_tags = error['pred']
                            tgt_tags = error['tgt']
                            self.errors['words'].append(words)
                            self.errors['pred'].append(pred_tags)
                            self.errors['tgt'].append(tgt_tags)

                        if self.has_c and len(self.metrics)>1:
                            p_pred = pred_dict['pred']
                            p_metrics({'pred': p_pred}, {'target': batch_y['p'], 'seq_len': batch_y['seq_len']})
                            c_metrics = self.metrics[1]
                            c_pred = pred_dict['c_pred']
                            c_metrics({'pred': c_pred}, {'target': batch_y['c'], 'seq_len': batch_y['seq_len']})
                        else:
                            p_metrics(pred_dict, {'target': batch_y['p'], 'seq_len': batch_y['seq_len']})

                        if self.use_tqdm:
                            pbar.update()

                        for metric in self.metrics:
                            eval_result = metric.get_metric()
                            if not isinstance(eval_result, dict):
                                raise TypeError(f"The return value of {_get_func_signature(metric.get_metric)} must be "
                                                f"`dict`, got {type(eval_result)}")
                            metric_name = metric.get_metric_name()
                            eval_results[metric_name] = eval_result


                    pbar.close()
                    end_time = time.time()
                    test_str = f'Evaluate data in {round(end_time - start_time, 2)} seconds!'
                    if self.verbose >= 0:
                        self.logger.info(test_str)
        except _CheckError as e:
            prev_func_signature = _get_func_signature(self._predict_func)
            _check_loss_evaluate(prev_func_signature=prev_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, pred_dict=pred_dict, target_dict=batch_y,
                                 dataset=self.data, check_level=0)
        finally:
            self._mode(network, is_test=False)
        if self.verbose >= 1:
            logger.info("[tester] \n{}".format(self._format_eval_results(eval_results)))
        return eval_results