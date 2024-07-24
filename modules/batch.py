
import atexit
import abc

from numbers import Number
import numpy as np
import torch
import torch.utils.data
from collections import defaultdict

from fastNLP import DataSet
from fastNLP import SequentialSampler, Sampler
import copy
import random


_python_is_exit = False


def _set_python_is_exit():
    global _python_is_exit
    _python_is_exit = True


atexit.register(_set_python_is_exit)


def _pad(batch_dict, dataset, as_numpy):
    result = {}
    for n, vlist in batch_dict.items():
        f = dataset.field_arrays[n]
        if f.padder is None:
            result[n] = np.array(vlist)
        else:
            res = f.pad(vlist)
            if not as_numpy:
                res, _ = _to_tensor(res, field_dtype=f.dtype)
            result[n] = res

    return result


class DataSetGetter:
    r"""
    传递给torch.utils.data.DataLoader获取数据，DataLoder会传入int的idx获取数据(调用这里的__getitem__()函数)。
    """
    def __init__(self, dataset: DataSet, as_numpy=False):
        self.dataset = dataset
        self.as_numpy = as_numpy
        self.idx_list = list(range(len(dataset)))

        self.x_names = {n for n, f in dataset.get_all_fields().items() if f.is_input}
        self.y_names = {n for n, f in dataset.get_all_fields().items() if f.is_target}

    def __getitem__(self, idx: int):
        # mapping idx to sampled idx
        idx = self.idx_list[idx]
        ins = self.dataset[idx]
        return idx, ins

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, ins_list: list):
        r"""

        :param batch: [[idx1, x_dict1, y_dict1], [idx2, x_dict2, y_dict2], [xx, xx, xx]]
        :return:
        """
        indices = []
        sin_x, sin_y = defaultdict(list), defaultdict(list)
        # 收集需要关注的field的数据
        for idx, ins in ins_list:
            indices.append(idx)
            for n, v in ins.items():
                if n in self.x_names:
                    sin_x[n].append(v)
                if n in self.y_names:
                    sin_y[n].append(v)
        # 根据情况，进行pad
        sin_x = _pad(sin_x, dataset=self.dataset, as_numpy=self.as_numpy)
        sin_y = _pad(sin_y, dataset=self.dataset, as_numpy=self.as_numpy)

        if not self.dataset.collater.is_empty():
            bx, by = self.dataset._collate_batch(ins_list)
            sin_x.update(bx)
            sin_y.update(by)

        return indices, sin_x, sin_y

    def __getattr__(self, item):
        if hasattr(self.dataset, item):
            return getattr(self.dataset, item)
        else:
            raise AttributeError("'DataSetGetter' object has no attribute '{}'".format(item))


class SamplerAdapter(torch.utils.data.Sampler):
    r"""
    用于传入torch.utils.data.DataLoader中，DataLoader会调用__iter__()方法获取index(一次只取一个int)

    """
    def __init__(self, sampler, dataset):
        super().__init__(dataset)
        self.sampler = sampler
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.sampler(self.dataset))


class MultiBatchIter:
    r"""
    Trainer用于迭代数据的类。继承该类，并实现get_num_batches(), get_batch_indices(), num_batches(), __iter__()方法以及dataset属性。

    """
    def __init__(self, datasets, batch_size=1, sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None,
                 batch_sampler=None):

        self.batch_sampler = batch_sampler
        self.dataiters = []

        for index, dataset in enumerate(datasets):
            # DataLoader的collate_fn输入是List[]，里面的元素是dataset[index]返回的结果
            if isinstance(sampler, Sampler):  # 如果时fastNLP的sampler需要adapt一下
                sampler = SamplerAdapter(sampler=sampler or SequentialSampler(), dataset=dataset)
            if collate_fn is None:
                # pytoch <= 1.1 中不能设置collate_fn=None
                dataiter = torch.utils.data.DataLoader(
                    dataset=dataset, batch_size=batch_size, sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory, drop_last=drop_last,
                    timeout=timeout, worker_init_fn=worker_init_fn,
                    batch_sampler=batch_sampler)
            else:
                dataiter = torch.utils.data.DataLoader(
                    dataset=dataset, batch_size=batch_size, sampler=sampler,
                    collate_fn=collate_fn[index], num_workers=num_workers,
                    pin_memory=pin_memory, drop_last=drop_last,
                    timeout=timeout, worker_init_fn=worker_init_fn,
                    batch_sampler=batch_sampler)
            self.dataiters.append(dataiter)

        # 以sampler的数量为准，因为DistributedSampler的时候每个进程上并不是所有的数据都用上了
        if self.batch_sampler is None:
            self._num_batches = 0
            for dataiter in self.dataiters:
                self._num_batches += self.get_num_batches(len(dataiter.sampler), batch_size, drop_last)
        else:
            self._num_batches = len(self.batch_sampler)
        self.batch_size = batch_size
        self.cur_batch_indices = None
        
        self.batch_list = torch.randperm(self._num_batches)
        self.batches_x = []
        self.batches_y = []
        self.batch_indices = []
        for dataiter in self.dataiters:
            for indices, batch_x, batch_y in dataiter:
                if 'chunk' in batch_x.keys():
                    batch_x['on_wsj'] = False
                else:
                    batch_x['on_wsj'] = True
                self.batch_indices.append(indices)
                batch_x_cp = copy.deepcopy(batch_x)  
                del batch_x
                batch_y_cp = copy.deepcopy(batch_y)  
                del batch_y
                self.batches_x.append(batch_x_cp)
                self.batches_y.append(batch_y_cp)

    @property
    def num_batches(self):
        return self._num_batches

    @num_batches.setter
    def num_batches(self, value):
        self._num_batches = value

    def init_iter(self):
        pass

    @staticmethod
    def get_num_batches(num_samples, batch_size, drop_last):
        r"""
        计算batch的数量。用于前端显示进度

        :param int num_samples:
        :param int batch_size:
        :param bool drop_last: 如果最后一个batch没有batch_size这么多，是否就丢掉。
        :return:
        """
        num_batches = num_samples // batch_size
        if not drop_last and (num_samples % batch_size > 0):
            num_batches += 1
        return num_batches

    def get_batch_indices(self):
        r"""
        获取最近输出的batch的index。用于溯源当前batch的数据

        :return:
        """
        return self.cur_batch_indices

    def __len__(self):
        return self.num_batches

    @property
    def datasets(self):
        r"""
        获取正在参与iterate的dataset

        :return:
        """
        datasets = []
        for dataiter in self.dataiters:
            datasets.append(dataiter.dataset)
        return datasets

    @abc.abstractmethod
    def __iter__(self):
        r"""
        用于实际数据循环的类，返回值需要为两个dict, 第一个dict中的内容会认为是input, 第二个dict中的内容会认为是target

        :return:
        """
        raise NotImplemented


class MultiDataSetIter(MultiBatchIter):
    r"""
    DataSetIter 用于从 `DataSet` 中按一定的顺序, 依次按 ``batch_size`` 的大小将数据取出，通过使用DataSetIter，可以不需要考虑
        输入的padding(由DataSet中每列的Padder决定了)以及不需要考虑将数据转为tensor。
    组成 `x` 和 `y`::

        batch = DataSetIter(data_set, batch_size=16, sampler=SequentialSampler())
        num_batch = len(batch)
        for batch_x, batch_y in batch:
            # do stuff ...

    """
    def __init__(self, datasets, batch_size=1, sampler=None, as_numpy=False, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None, batch_sampler=None):
        datasets_reformed = []
        collate_fn = []
        for dataset in datasets:
            assert isinstance(dataset, DataSet)
            d = DataSetGetter(dataset, as_numpy)
            datasets_reformed.append(d)
            collate_fn.append(d.collate_fn)
        if batch_sampler is not None:
            batch_size = 1
            sampler = None
            drop_last = False
        super().__init__(
            datasets=datasets_reformed, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
            collate_fn=collate_fn, batch_sampler=batch_sampler
        )
        

    def __iter__(self):
        self.init_iter()
        for index in self.batch_list:
            self.cur_batch_indices = self.batch_indices[index]
            batch_x = self.batches_x[index]
            batch_y = self.batches_y[index]
            yield batch_x, batch_y


class CLBatchIter:
    def __init__(self, datasets, batch_size=1, cl_size=4, sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None,
                 batch_sampler=None, skip=False):

        self.batch_sampler = batch_sampler
        self.dataiters = []
        self.conll = datasets[0]
        if skip:
            datasets = [datasets[1]]
        self.batches_x = []
        self.batches_y = []
        self.cl_size = cl_size
        self.batch_size= batch_size

        for index, dataset in enumerate(datasets):
            # DataLoader的collate_fn输入是List[]，里面的元素是dataset[index]返回的结果
            if isinstance(sampler, Sampler):  # 如果时fastNLP的sampler需要adapt一下
                sampler = SamplerAdapter(sampler=sampler or SequentialSampler(), dataset=dataset)
            if collate_fn is None:
                # pytoch <= 1.1 中不能设置collate_fn=None
                dataiter = torch.utils.data.DataLoader(
                    dataset=dataset, batch_size=batch_size, sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory, drop_last=drop_last,
                    timeout=timeout, worker_init_fn=worker_init_fn,
                    batch_sampler=batch_sampler)
            else:
                dataiter = torch.utils.data.DataLoader(
                    dataset=dataset, batch_size=batch_size, sampler=sampler,
                    collate_fn=collate_fn[index], num_workers=num_workers,
                    pin_memory=pin_memory, drop_last=drop_last,
                    timeout=timeout, worker_init_fn=worker_init_fn,
                    batch_sampler=batch_sampler)
            self.dataiters.append(dataiter)

        # 以sampler的数量为准，因为DistributedSampler的时候每个进程上并不是所有的数据都用上了
        if self.batch_sampler is None:
            self._num_batches = 0
            for dataiter in self.dataiters:
                self._num_batches += self.get_num_batches(len(dataiter.sampler), batch_size, drop_last)
        else:
            self._num_batches = len(self.batch_sampler)
        self.batch_size = batch_size
        self.cur_batch_indices = None
        self.batch_indices = torch.arange(self._num_batches)

    @property
    def num_batches(self):
        return self._num_batches

    @num_batches.setter
    def num_batches(self, value):
        self._num_batches = value

    def init_iter(self):
        self.batches_x = []
        self.batches_y = []
        for dataiter in self.dataiters:
            for indices, batch_x, batch_y in dataiter:
                if 'chunk' in batch_x.keys():
                    batch_x['on_wsj'] = False
                    batch_x['cl_samples'] = None
                else:
                    batch_x['on_wsj'] = True
                    max_len = batch_x['target'].size()[1]
                    cl_idx = random.sample(range(self.conll.get_length()), self.cl_size*self.batch_size)
                    cl_samples = [self.conll.__getitem__(idx) for idx in cl_idx]
                    _, sin_x, _ = self.conll.collate_fn(cl_samples)
                    sin_len = sin_x['target'].size()[1]
                    if sin_len < max_len:
                        padding = torch.zeros((self.cl_size*self.batch_size, max_len-sin_len),dtype=torch.int64)
                        sin_x['words'] = torch.cat((sin_x['words'],padding),dim=1)  #comment out if Flair
                        sin_x['chunk'] = torch.cat((sin_x['chunk'],padding),dim=1)
                    elif sin_len > max_len:
                        sin_x['words'], _ = torch.split(sin_x['words'],[max_len, sin_len-max_len],dim=1)  #comment out if Flair
                        sin_x['chunk'], _ = torch.split(sin_x['chunk'],[max_len, sin_len-max_len],dim=1)
                    ws = batch_x['target'].size()[0]
                    bz = sin_x['target'].size()[0]//self.cl_size
                    if bz > ws:
                        sin_x['words'] = sin_x['words'][:ws*cl_size]
                        sin_x['chunk'], _ = torch.split(sin_x['chunk'],[ws*self.cl_size, (bz-ws)*self.cl_size],dim=0)
                    sin_x.pop('target', None)
                    sin_x.pop('seq_len', None)
                    batch_x['cl_samples'] = sin_x
                # self.batch_indices.append(indices)
                batch_x_cp = copy.deepcopy(batch_x)  
                del batch_x
                batch_y_cp = copy.deepcopy(batch_y)  
                del batch_y
                self.batches_x.append(batch_x_cp)
                self.batches_y.append(batch_y_cp)

    @staticmethod
    def get_num_batches(num_samples, batch_size, drop_last):
        r"""
        计算batch的数量。用于前端显示进度

        :param int num_samples:
        :param int batch_size:
        :param bool drop_last: 如果最后一个batch没有batch_size这么多，是否就丢掉。
        :return:
        """
        num_batches = num_samples // batch_size
        if not drop_last and (num_samples % batch_size > 0):
            num_batches += 1
        return num_batches

    def get_batch_indices(self):
        r"""
        获取最近输出的batch的index。用于溯源当前batch的数据

        :return:
        """
        return self.cur_batch_indices

    def __len__(self):
        return self.num_batches

    @property
    def datasets(self):
        r"""
        获取正在参与iterate的dataset

        :return:
        """
        datasets = []
        for dataiter in self.dataiters:
            datasets.append(dataiter.dataset)
        return datasets

    @abc.abstractmethod
    def __iter__(self):
        r"""
        用于实际数据循环的类，返回值需要为两个dict, 第一个dict中的内容会认为是input, 第二个dict中的内容会认为是target

        :return:
        """
        raise NotImplemented


class CLBatchIter2:
    def __init__(self, datasets, batch_size=1, cl_size=4, sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None,
                 batch_sampler=None):

        self.batch_sampler = batch_sampler
        self.dataiters = []
        self.conll = datasets[0]
        self.wsj = datasets[1]

        for index, dataset in enumerate(datasets):
            # DataLoader的collate_fn输入是List[]，里面的元素是dataset[index]返回的结果
            if isinstance(sampler, Sampler):  # 如果时fastNLP的sampler需要adapt一下
                sampler = SamplerAdapter(sampler=sampler or SequentialSampler(), dataset=dataset)
            if collate_fn is None:
                # pytoch <= 1.1 中不能设置collate_fn=None
                dataiter = torch.utils.data.DataLoader(
                    dataset=dataset, batch_size=batch_size, sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory, drop_last=drop_last,
                    timeout=timeout, worker_init_fn=worker_init_fn,
                    batch_sampler=batch_sampler)
            else:
                dataiter = torch.utils.data.DataLoader(
                    dataset=dataset, batch_size=batch_size, sampler=sampler,
                    collate_fn=collate_fn[index], num_workers=num_workers,
                    pin_memory=pin_memory, drop_last=drop_last,
                    timeout=timeout, worker_init_fn=worker_init_fn,
                    batch_sampler=batch_sampler)
            self.dataiters.append(dataiter)

        # 以sampler的数量为准，因为DistributedSampler的时候每个进程上并不是所有的数据都用上了
        if self.batch_sampler is None:
            self._num_batches = 0
            for dataiter in self.dataiters:
                self._num_batches += self.get_num_batches(len(dataiter.sampler), batch_size, drop_last)
        else:
            self._num_batches = len(self.batch_sampler)
        self.batch_size = batch_size
        self.cur_batch_indices = None
        
        self.batch_list = torch.randperm(self._num_batches)
        self.batches_x = []
        self.batches_y = []
        self.batch_indices = []
        for dataiter in self.dataiters:
            for indices, batch_x, batch_y in dataiter:
                if 'chunk' in batch_x.keys():
                    batch_x['on_wsj'] = False
                    batch_x['cl_samples'] = None
                    max_len = batch_x['target'].size()[1]
                    cl_idx = random.sample(range(self.wsj.get_length()), cl_size*batch_size)
                    cl_samples = [self.wsj.__getitem__(idx) for idx in cl_idx]
                    _, sin_x, _ = self.wsj.collate_fn(cl_samples)
                    sin_len = sin_x['target'].size()[1]
                    if sin_len < max_len:
                        padding = torch.zeros((cl_size*batch_size, max_len-sin_len),dtype=torch.int64)
                        sin_x['words'] = torch.cat((sin_x['words'],padding),dim=1)
                        sin_x['target'] = torch.cat((sin_x['target'],padding),dim=1)
                    elif sin_len > max_len:
                        sin_x['words'], _ = torch.split(sin_x['words'],[max_len, sin_len-max_len],dim=1)
                        sin_x['target'], _ = torch.split(sin_x['target'],[max_len, sin_len-max_len],dim=1)
                    ws = batch_x['target'].size()[0]
                    bz = sin_x['target'].size()[0]//cl_size
                    if bz > ws:
                        sin_x['words'] = sin_x['words'][:ws*cl_size]
                        sin_x['target'], _ = torch.split(sin_x['target'],[ws*cl_size, (bz-ws)*cl_size],dim=0)
                    sin_x.pop('seq_len', None)
                    batch_x['cl_samples'] = sin_x
                else:
                    batch_x['on_wsj'] = True
                    max_len = batch_x['target'].size()[1]
                    cl_idx = random.sample(range(self.conll.get_length()), cl_size*batch_size)
                    cl_samples = [self.conll.__getitem__(idx) for idx in cl_idx]
                    _, sin_x, _ = self.conll.collate_fn(cl_samples)
                    sin_len = sin_x['target'].size()[1]
                    if sin_len < max_len:
                        padding = torch.zeros((cl_size*batch_size, max_len-sin_len),dtype=torch.int64)
                        sin_x['words'] = torch.cat((sin_x['words'],padding),dim=1)
                        sin_x['chunk'] = torch.cat((sin_x['chunk'],padding),dim=1)
                    elif sin_len > max_len:
                        sin_x['words'], _ = torch.split(sin_x['words'],[max_len, sin_len-max_len],dim=1)
                        sin_x['chunk'], _ = torch.split(sin_x['chunk'],[max_len, sin_len-max_len],dim=1)
                    ws = batch_x['target'].size()[0]
                    bz = sin_x['target'].size()[0]//cl_size
                    if bz > ws:
                        sin_x['words'] = sin_x['words'][:ws*cl_size]
                        sin_x['chunk'], _ = torch.split(sin_x['chunk'],[ws*cl_size, (bz-ws)*cl_size],dim=0)
                    sin_x.pop('target', None)
                    sin_x.pop('seq_len', None)
                    batch_x['cl_samples'] = sin_x
                self.batch_indices.append(indices)
                batch_x_cp = copy.deepcopy(batch_x)  
                del batch_x
                batch_y_cp = copy.deepcopy(batch_y)  
                del batch_y
                self.batches_x.append(batch_x_cp)
                self.batches_y.append(batch_y_cp)

    @property
    def num_batches(self):
        return self._num_batches

    @num_batches.setter
    def num_batches(self, value):
        self._num_batches = value

    def init_iter(self):
        pass

    @staticmethod
    def get_num_batches(num_samples, batch_size, drop_last):
        r"""
        计算batch的数量。用于前端显示进度

        :param int num_samples:
        :param int batch_size:
        :param bool drop_last: 如果最后一个batch没有batch_size这么多，是否就丢掉。
        :return:
        """
        num_batches = num_samples // batch_size
        if not drop_last and (num_samples % batch_size > 0):
            num_batches += 1
        return num_batches

    def get_batch_indices(self):
        r"""
        获取最近输出的batch的index。用于溯源当前batch的数据

        :return:
        """
        return self.cur_batch_indices

    def __len__(self):
        return self.num_batches

    @property
    def datasets(self):
        r"""
        获取正在参与iterate的dataset

        :return:
        """
        datasets = []
        for dataiter in self.dataiters:
            datasets.append(dataiter.dataset)
        return datasets

    @abc.abstractmethod
    def __iter__(self):
        r"""
        用于实际数据循环的类，返回值需要为两个dict, 第一个dict中的内容会认为是input, 第二个dict中的内容会认为是target

        :return:
        """
        raise NotImplemented




class CLDataSetIter(CLBatchIter):
    r"""
    DataSetIter 用于从 `DataSet` 中按一定的顺序, 依次按 ``batch_size`` 的大小将数据取出，通过使用DataSetIter，可以不需要考虑
        输入的padding(由DataSet中每列的Padder决定了)以及不需要考虑将数据转为tensor。
    组成 `x` 和 `y`::

        batch = DataSetIter(data_set, batch_size=16, sampler=SequentialSampler())
        num_batch = len(batch)
        for batch_x, batch_y in batch:
            # do stuff ...

    """
    def __init__(self, datasets, batch_size=1, cl_size=4, sampler=None, as_numpy=False, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None, batch_sampler=None, skip=False):
        datasets_reformed = []
        collate_fn = []
        for dataset in datasets:
            assert isinstance(dataset, DataSet)
            d = DataSetGetter(dataset, as_numpy)
            datasets_reformed.append(d)
            collate_fn.append(d.collate_fn)
        if batch_sampler is not None:
            batch_size = 1
            sampler = None
            drop_last = False
        super().__init__(
            datasets=datasets_reformed, batch_size=batch_size, cl_size=cl_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
            collate_fn=collate_fn, batch_sampler=batch_sampler,skip=skip
        )
        

    def __iter__(self):
        self.init_iter()
        batch_list = torch.randperm(self._num_batches)
        for index in batch_list:
            self.cur_batch_indices = self.batch_indices[index]
            batch_x = self.batches_x[index]
            batch_y = self.batches_y[index]
            yield batch_x, batch_y




def _to_tensor(batch, field_dtype):
    r"""

    :param batch: np.array()
    :param field_dtype: 数据类型
    :return: batch, flag. 如果传入的数据支持转为tensor，返回的batch就是tensor，且flag为True；如果传入的数据不支持转为tensor，
        返回的batch就是原来的数据，且flag为False
    """
    try:
        if field_dtype is not None and isinstance(field_dtype, type)\
                and issubclass(field_dtype, Number) \
                and not isinstance(batch, torch.Tensor):
            new_batch = torch.as_tensor(batch)
            flag = True
        else:
            new_batch = batch
            flag = False
        if torch.is_tensor(new_batch):
            if 'float' in new_batch.dtype.__repr__():
                new_batch = new_batch.float()
            elif 'int' in new_batch.dtype.__repr__():
                new_batch = new_batch.long()
        return new_batch, flag
    except Exception as e:
        raise e




