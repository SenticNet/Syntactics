from fastNLP import Trainer, DataSet, DataSetIter, Tester
from fastNLP import Optimizer, Sampler, RandomSampler
from fastNLP import Callback
from .multitask_tester import Multitask_Tester
from .batch import MultiDataSetIter, CLDataSetIter
from .utils import _CheckError
from .utils import _build_args
from .utils import _check_loss_evaluate
from .utils import _move_dict_value_to_device
from .utils import _get_func_signature
from .utils import _get_model_device
from .utils import _move_model_to_device
from ._parallel_utils import _data_parallel_wrapper
from ._parallel_utils import _model_contains_inner_module
from functools import partial

from .multitask_tester import Multitask_Tester

import time
from datetime import datetime, timedelta

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except:
    from .utils import _pseudo_tqdm as tqdm


class Multitask_Trainer(Trainer):
    def __init__(self, train_data, model, optimizer=None, loss=None, add_data=None,
                 batch_size=32, sampler=None, pin_memory=True, drop_last=True, update_every=1,
                 num_workers=0, n_epochs=10, print_every=5,
                 dev_data=None, test_data=None, metrics=None, c_metrics=None, metric_key=None,
                 validate_every=-1, save_path=None, use_tqdm=True, device=None,
                 callbacks=None, check_code_level=0, logger_path=None, **kwargs):
        Trainer.__init__(self, train_data, model, optimizer, loss, batch_size,
                         sampler, drop_last, update_every, num_workers, n_epochs,
                         print_every, dev_data, metrics, metric_key, validate_every,
                         save_path, use_tqdm, device, callbacks, check_code_level, **kwargs)

        if add_data is not None:
            if model.cl_size is None:
                self.data_iterator = MultiDataSetIter(datasets=[train_data,add_data], batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
            else:
                self.data_iterator = CLDataSetIter(datasets=[train_data,add_data], batch_size=batch_size, cl_size=model.cl_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
            self.n_steps = len(self.data_iterator) * self.n_epochs
            self.baseline = False
        else:
            self.baseline = True
            self.data_iterator_add = None
        self.c_metrics = c_metrics
        self.logger_path = logger_path
        self.test_data = test_data
        if test_data is not None:
            self.best_p_indicator = None
            self.best_c_indicator = None
            self.p_tester = Tester(model=self.model,
                                 data=self.test_data[0],
                                 metrics=self.metrics,
                                 batch_size=self.batch_size,
                                 device=None, 
                                 verbose=0,
                                 use_tqdm=self.test_use_tqdm,
                                 sampler=None)
            self.c_tester = Multitask_Tester(model=self.model,
                                 data=self.test_data[1],
                                 metrics=self.c_metrics,
                                 batch_size=self.batch_size,
                                 device=None, 
                                 verbose=0,
                                 use_tqdm=self.test_use_tqdm,
                                 sampler=None)
            
    def setBaseline(self, v):
        self.baseline = v


    def _data_forward(self, network, x):
        x = _build_args(self._forward_func, **x)
        y = network(**x)
        if not isinstance(y, dict):
            raise TypeError(
                f"The return value of {_get_func_signature(self._forward_func)} should be dict, got {type(y)}.")
        return y

    def _do_p_test(self, epoch, step):
        # self.callback_manager.on_valid_begin()
        res = self.p_tester.test()
        is_better_eval = False
        indicator, indicator_val = _check_eval_results(res, 'acc', self.metrics)
        if self.best_p_indicator is None:
            self.best_p_indicator = indicator_val
        else:
            if indicator_val > self.best_p_indicator:
                self.best_p_indicator = indicator_val
                self.best_p_perf = res
                self.best_p_epoch = epoch
                is_better_eval = True
        return res

    def _do_c_test(self, epoch, step):
        # self.callback_manager.on_valid_begin()
        res = self.c_tester.test()
        is_better_eval = False
        indicator, indicator_val = _check_eval_results(res, 'f', self.c_metrics)
        if self.best_c_indicator is None:
            self.best_c_indicator = indicator_val
        else:
            if indicator_val > self.best_c_indicator:
                self.best_c_indicator = indicator_val
                self.best_c_perf = res
                self.best_c_epoch = epoch
                is_better_eval = True
        return res
        

    def train(self, load_best_model=False, on_exception='auto', **kwargs):
        results = {}
        verbose = kwargs.get('verbose', 0)
        if self.n_epochs <= 0:
            self.logger.info(f"training epoch is {self.n_epochs}, nothing was done.")
            results['seconds'] = 0.
            return results
        try:
            self._model_device = _get_model_device(self.model)
            self._mode(self.model, is_test=False)
            self._load_best_model = load_best_model
            self.start_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
            start_time = time.time()
            self.logger.info("training epochs started " + self.start_time)
            self.step = 0
            self.epoch = 1
            try:
                self.callback_manager.on_train_begin()
                self._train()
                self.callback_manager.on_train_end()

            except BaseException as e:
                self.callback_manager.on_exception(e)
                if verbose>0:
                    self.logger.info(f"The data indices for current batch are: {self.data_iterator.cur_batch_indices}.")
                if on_exception == 'auto':
                    if not isinstance(e, (CallbackException, KeyboardInterrupt)):
                        raise e
                elif on_exception == 'raise':
                    raise e

        finally:
            if self.dev_data is not None and self.best_dev_perf is not None:
                dev_str = "\nIn Epoch {}, got best dev performance: ".format(self.best_dev_epoch)
                results['best_eval'] = self.best_dev_perf
                results['best_eval_epoch'] = self.best_dev_epoch
                results['best_eval_step'] = self.best_dev_step

            if self.test_data is not None:
                if self.best_p_perf is not None:
                    p_test_str = "\nIn Epoch {}, got best p_test performance: \n".format(self.best_p_epoch)
                    results['best_p_test'] = self.best_p_perf
                    results['best_p_epoch'] = self.best_p_epoch
                if self.best_c_perf is not None:
                    con_test_str = "\nIn Epoch {}, got best c_test performance: \n".format(self.best_c_epoch)
                    results['best_c_test'] = self.best_c_perf
                    results['best_c_epoch'] = self.best_c_epoch
                    
            with open(self.logger_path, 'a') as f: 
                f.write(dev_str)
                f.write(self.tester._format_eval_results(self.best_dev_perf))
                f.write(p_test_str)
                f.write(self.p_tester._format_eval_results(self.best_p_perf))
                f.write(con_test_str)
                f.write(self.c_tester._format_eval_results(self.best_c_perf))
                f.write('\n\n')
                f.close()

        results['seconds'] = round(time.time() - start_time, 2)
        
        return results


    def _train(self, load_best_model=False, on_exception='auto'):
        if not self.use_tqdm:
            from .utils import _pseudo_tqdm as inner_tqdm
        else:
            inner_tqdm = tqdm
        start = time.time()
        with inner_tqdm(total=self.n_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                        initial=self.step) as pbar:
            self.pbar = pbar
            p_avg_loss = 0
            c_avg_loss = 0
            c_steps = 0
            self.batch_per_epoch = self.data_iterator.num_batches
            for epoch in range(self.epoch, self.n_epochs + 1):
                data_iterator = self.data_iterator
                self.epoch = epoch
                pbar.set_description_str(
                    desc="Epoch {}/{}".format(epoch, self.n_epochs))
                self.callback_manager.on_epoch_begin()
                for batch_x, batch_y in data_iterator:
                    self.step += 1
                    _move_dict_value_to_device(
                        batch_x, batch_y, device=self._model_device)
                    indices = data_iterator.get_batch_indices()
                    self.callback_manager.on_batch_begin(
                        batch_x, batch_y, indices)
                    prediction = self._data_forward(self.model, batch_x)

                    self.callback_manager.on_loss_begin(batch_y, prediction)
                    
                    loss = self._compute_loss(prediction, batch_y).mean()
                    if batch_x['on_np']:
                        p_avg_loss += loss.item()
                    else:
                        c_steps += 1
                        p_loss = prediction['p_loss'].mean()
                        c_loss = prediction['c_loss'].mean()
                        p_avg_loss += p_loss.item()
                        c_avg_loss += c_loss.item()

                    # Is loss NaN or inf? requires_grad = False
                    self.callback_manager.on_backward_begin(loss)
                    self._grad_backward(loss)
                    self.callback_manager.on_backward_end()

                    self._update()
                    self.callback_manager.on_step_end()

                    if self.step % self.print_every == 0:
                        p_avg_loss = float(p_avg_loss) / self.print_every
                        if c_steps > 0:
                            c_avg_loss = float(c_avg_loss) / c_steps
                        if self.use_tqdm:
                            print_output = "p_loss:{:<6.5f} c_loss:{:<6.5f}".format(p_avg_loss, c_avg_loss)
                            pbar.update(self.print_every)
                        else:
                            end = time.time()
                            diff = timedelta(seconds=round(end - start))
                            print_output = "[epoch: {:>3} step: {:>4}] train loss: {:>4.6} time: {}".format(
                                epoch, self.step, p_avg_loss, diff)
                        pbar.set_postfix_str(print_output)
                        p_avg_loss = 0
                        c_avg_loss = 0
                        c_steps = 0
                    self.callback_manager.on_batch_end()

                # ================= mini-batch end ==================== #
                # begin_str = "Epoch {}. Training completed in {}s. \n".format(epoch, round(time.time() - start_time, 2))
                begin_str = "Epoch {}. Time: {} \n".format(epoch, datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
                # self.logger.info(train_time_str)
                
                if self.validate_every < 0 and self.dev_data is not None:
                    eval_res = self._do_validation(epoch=epoch, step=self.step)

                if self.test_data is not None:
                    p_test_res = self._do_p_test(epoch=epoch, step=self.step)
                    c_test_res = self._do_c_test(epoch=epoch, step=self.step)
                        
                with open(self.logger_path, 'a') as f: 
                        f.write(begin_str)
                        f.write('p_dev: \n'+ self.tester._format_eval_results(eval_res) + '\n')
                        f.write('p_test: \n' + self.p_tester._format_eval_results(p_test_res) + '\n')
                        f.write('c_test: \n' + self.c_tester._format_eval_results(c_test_res) + '\n\n')
                        f.close()

                # lr decay; early stopping
                self.callback_manager.on_epoch_end()
                torch.cuda.empty_cache()
            # =============== epochs end =================== #
            pbar.close()
            self.pbar = None
        # ============ tqdm end ============== #

 
 
def _check_eval_results(metrics, metric_key, metric_list):
    if isinstance(metrics, tuple):
        loss, metrics = metrics
    
    if isinstance(metrics, dict):
        metric_dict = list(metrics.values())[0]  
        
        if metric_key is None or metric_key not in metric_dict:
            indicator_val, indicator = list(metric_dict.values())[0], list(metric_dict.keys())[0]
        else:
            indicator_val = metric_dict[metric_key]
            indicator = metric_key
    else:
        raise RuntimeError("Invalid metrics type. Expect {}, got {}".format((tuple, dict), type(metrics)))
    return indicator, indicator_val