
from fastNLP.io import Pipe, ConllLoader
from fastNLP.io import DataBundle
from fastNLP.io.pipe.utils import _add_words_field
from fastNLP.io.pipe.utils import iob2, iob2bioes
from fastNLP.io.pipe.utils import _add_chars_field
from fastNLP.io.utils import check_loader_paths
from fastNLP import Vocabulary, DataSet, Instance
from fastNLP.io import Conll2003NERLoader
from fastNLP import Const
import re
import copy
from ._logger import logger

import torch
import torch.nn as nn
# from flair.embeddings import FlairEmbeddings, WordEmbeddings, StackedEmbeddings
# import flair
# from flair.data import Sentence
# from flair.tokenization import SpacyTokenizer

def word_shape(words):
    shapes = []
    for word in words:
        caps = []
        for char in word:
            caps.append(char.isupper())
        if all(caps):
            shapes.append(0)
        elif any(caps) is False:
            shapes.append(1)
        elif caps[0]:
            shapes.append(2)
        elif any(caps):
            shapes.append(3)
        else:
            shapes.append(4)
    return shapes

class MultitaskPipe(Pipe):

    def __init__(self, chunk_encoding_type='bioes', lower: bool = False):

        if chunk_encoding_type == 'bio':
            self.chunk_convert_tag = iob2
        elif chunk_encoding_type == 'bioes':
            self.chunk_convert_tag = lambda tags: iob2bioes(iob2(tags))
        else:
            raise ValueError("chunk_encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
    
    def process(self, data_bundle) -> DataBundle:
        c_datasets = []
        for name, dataset in data_bundle.datasets.items():
            if dataset.has_field('c'):
                dataset.drop(lambda x: "-DOCSTART-" in x[Const.RAW_WORD])
                dataset.apply_field(self.chunk_convert_tag, field_name='c', new_field_name='c')
                c_datasets.append(dataset)
        
        _add_words_field(data_bundle, lower=self.lower)
        
        # index
        _indexize(data_bundle, input_field_names=Const.INPUT, target_field_names=['p'])
        
        tgt_vocab = Vocabulary(unknown=None, padding=None)
        tgt_vocab.from_dataset(*c_datasets, field_name='c')
        tgt_vocab.index_dataset(*c_datasets, field_name='c')
        data_bundle.set_vocab(tgt_vocab, 'c')
        
        input_fields = [Const.INPUT, Const.INPUT_LEN, 'p', 'c']
        target_fields = ['p', 'c', Const.INPUT_LEN]
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
        
        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)
        
        return data_bundle
    
    
    def process_from_file(self, paths):
        data_bundle = ConllLoader(headers=['raw_words', 'p', 'c']).load(paths[0])
        pos_bundle = ConllLoader(headers=['raw_words', 'p']).load(paths[1])
        for name, dataset in pos_bundle.datasets.items():
            data_bundle.set_dataset(dataset, 'p_'+name)
        return self.process(data_bundle)


class MultitaskFlairPipe(Pipe):

    def __init__(self, chunk_encoding_type='bioes', lower: bool = False):

        if chunk_encoding_type == 'bio':
            self.chunk_convert_tag = iob2
        elif chunk_encoding_type == 'bioes':
            self.chunk_convert_tag = lambda tags: iob2bioes(iob2(tags))
        else:
            raise ValueError("chunk_encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
    
    def process(self, data_bundle) -> DataBundle:
        c_datasets = []
        for name, dataset in data_bundle.datasets.items():
            if dataset.has_field('c'):
                dataset.drop(lambda x: "-DOCSTART-" in x[Const.RAW_WORD])
                dataset.apply_field(self.chunk_convert_tag, field_name='c', new_field_name='c')
                c_datasets.append(dataset)
        
        _add_words_field(data_bundle, lower=self.lower)
        
        # index
        _indexize(data_bundle, input_field_names=Const.INPUT, target_field_names=['p'])
        
        tgt_vocab = Vocabulary(unknown=None, padding=None)
        tgt_vocab.from_dataset(*c_datasets, field_name='c')
        tgt_vocab.index_dataset(*c_datasets, field_name='c')
        data_bundle.set_vocab(tgt_vocab, 'c')
        
        input_fields = [Const.INPUT, Const.INPUT_LEN, 'raw_words','p', 'c']
        target_fields = ['p', 'c', Const.INPUT_LEN]
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
        
        data_bundle.rename_field('raw_words','words')
        data_bundle.set_ignore_type('words')
        # data_bundle.apply(self.toSentence,'raw_words')
            
        
        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)
        
        return data_bundle
        
    def toSentence(self,instance):
        sentence = instance['raw_words']
        words = Sentence(sentence)
        return words

    
    def process_from_file(self, paths):
        data_bundle = ConllLoader(headers=['raw_words', 'p', 'c']).load(paths[0])
        pos_bundle = ConllLoader(headers=['raw_words', 'p']).load(paths[1])
        for name, dataset in pos_bundle.datasets.items():
            data_bundle.set_dataset(dataset, 'p_'+name)
        return self.process(data_bundle)


class FlairPipe(Pipe):

    def __init__(self, chunk_encoding_type='bioes', lower: bool = False):

        if chunk_encoding_type == 'bio':
            self.chunk_convert_tag = iob2
        elif chunk_encoding_type == 'bioes':
            self.chunk_convert_tag = lambda tags: iob2bioes(iob2(tags))
        else:
            raise ValueError("chunk_encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
        
        self.embed = FlairEmbed()
        self.count = 0
        
    
    def process(self, data_bundle) -> DataBundle:
        chunk_datasets = []
        for name, dataset in data_bundle.datasets.items():
            if dataset.has_field('chunk'):
                dataset.drop(lambda x: "-DOCSTART-" in x[Const.RAW_WORD])
                dataset.apply_field(self.chunk_convert_tag, field_name='chunk', new_field_name='chunk')
                chunk_datasets.append(dataset)
        
        _add_words_field(data_bundle, lower=self.lower)
        
        # index
        _indexize(data_bundle, input_field_names=Const.INPUT, target_field_names=['pos'])
        
        tgt_vocab = Vocabulary(unknown=None, padding=None)
        tgt_vocab.from_dataset(*chunk_datasets, field_name='chunk')
        tgt_vocab.index_dataset(*chunk_datasets, field_name='chunk')
        data_bundle.set_vocab(tgt_vocab, 'chunk')
        
        input_fields = [Const.INPUT, Const.INPUT_LEN, Const.TARGET, 'chunk']
        target_fields = [Const.TARGET, 'chunk', Const.INPUT_LEN]
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
        
        data_bundle.set_ignore_type('words')
        data_bundle.apply(self.toSentence,'words')
        data_bundle.rename_field('pos','target')
            
        
        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)
        
        return data_bundle
        
    def toSentence(self,instance):
        self.count +=1
        if self.count%50==0:
            print(self.count)
        sentence = instance['raw_words']
        words = Sentence(sentence)
        # words.to(flair.device)
        words = self.embed(words)
        return words

    
    def process_from_file(self, paths):
        data_bundle = ConllLoader(headers=['raw_words', 'pos', 'chunk']).load(paths[0])
        pos_bundle = ConllLoader(headers=['raw_words', 'pos']).load(paths[1])
        for name, dataset in pos_bundle.datasets.items():
            data_bundle.set_dataset(dataset, 'pos_'+name)
        return self.process(data_bundle)



class Conll2000Pipe(Pipe):

    def __init__(self, chunk_encoding_type='bioes', lower: bool = False, vocab=None):

        if chunk_encoding_type == 'bio':
            self.chunk_convert_tag = iob2
        elif chunk_encoding_type == 'bioes':
            self.chunk_convert_tag = lambda tags: iob2bioes(iob2(tags))
        else:
            raise ValueError("chunk_encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
        self.vocab = vocab
    
    def process(self, data_bundle) -> DataBundle:
        for name, dataset in data_bundle.datasets.items():
            dataset.drop(lambda x: "-DOCSTART-" in x[Const.RAW_WORD])
            dataset.apply_field(self.chunk_convert_tag, field_name='chunk', new_field_name='chunk')
        
        _add_words_field(data_bundle, lower=self.lower)
        
        # index
        if self.vocab == None:
            _indexize(data_bundle, input_field_names=Const.INPUT, target_field_names=['pos'])
        else:
            _indexize(data_bundle, input_field_names=Const.INPUT, target_field_names=None)
            pos_vocab = self.vocab
            pos_vocab.index_dataset(*data_bundle.datasets.values(), field_name='pos') 
            data_bundle.set_vocab(pos_vocab, 'pos')

        tgt_vocab = Vocabulary(unknown=None, padding=None)
        tgt_vocab.from_dataset(*data_bundle.datasets.values(), field_name='chunk')
        tgt_vocab.index_dataset(*data_bundle.datasets.values(), field_name='chunk')
        data_bundle.set_vocab(tgt_vocab, 'chunk')
        
        input_fields = [Const.INPUT, Const.INPUT_LEN, 'pos', 'chunk']
        target_fields = ['pos', 'chunk', Const.INPUT_LEN]
        #target_fields = ['chunk', Const.INPUT_LEN]
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
        
        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)
        
        return data_bundle
    
    
    def process_from_file(self, paths):
        data_bundle = ConllLoader(headers=['raw_words', 'pos', 'chunk']).load(paths)
        return self.process(data_bundle)

    
    
    
    
class WSJPipe(Pipe):

    def __init__(self, lower: bool = False):

        self.lower = lower
        
    def process(self, data_bundle) -> DataBundle:
        for name, dataset in data_bundle.datasets.items():
            dataset.drop(lambda x: "-DOCSTART-" in x[Const.RAW_WORD])
        
        _add_words_field(data_bundle, lower=self.lower)
        
        _indexize(data_bundle, input_field_names=Const.INPUT, target_field_names=['pos'])
        
        input_fields = [Const.INPUT, Const.INPUT_LEN, 'pos']
        target_fields = ['pos', Const.INPUT_LEN]
        
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
        
        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)
        
        return data_bundle
    
    
    def process_from_file(self, paths):
        data_bundle = ConllLoader(headers=['raw_words', 'pos']).load(paths)
        return self.process(data_bundle)




class SentencesPipe(Pipe):
    def __init__(self, lower: bool = False, word_shape: bool=False):
        self.lower = lower
        self.word_shape = word_shape

    def process(self, data_bundle: DataBundle, vocab) -> DataBundle:
        _add_words_field(data_bundle, lower=self.lower)

        if self.word_shape:
            data_bundle.apply_field(word_shape, field_name='raw_words', new_field_name='word_shapes')
            data_bundle.set_input('word_shapes')

        vocab.index_dataset(*data_bundle.datasets.values(), field_name=Const.INPUT)

        input_fields = [Const.INPUT, Const.INPUT_LEN]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)

        data_bundle.set_input(*input_fields)

        return data_bundle


    def read_sentences(self, path, encoding='utf-8'):
        with open(path, 'r', encoding=encoding) as f:
            for line_idx, line in enumerate(f, 1):
                #line = re.sub(r'[^\w\s]','',line)
                line = line.strip()
                if line == '':
                    continue
                else:
                    try:
                        res = line.split()
                        yield line_idx, res
                    except Exception as e:
                        logger.error('invalid instance ends at line: {}'.format(line_idx))
                        raise e

    def process_from_file(self, path, vocab) -> DataBundle:
        ds = DataSet()
        for idx, data in self.read_sentences(path):
            ins = {'raw_words': data}
            ds.append(Instance(**ins))

        data_bundle = DataBundle(datasets = {'pred': ds})
        data_bundle = self.process(data_bundle, vocab)

        return data_bundle


class ChunkerPipe(SentencesPipe):

    def read_pos_tags(self, path, encoding='utf-8'):
        with open(path, 'r', encoding=encoding) as f:
            for line_idx, line in enumerate(f, 1):
                line = line.split()
                if line == '':
                    continue
                else:
                    try:
                        res = []
                        for tag in line:
                            res.append(int(tag))
                        yield line_idx, res
                    except Exception as e:
                        logger.error('invalid instance ends at line: {}'.format(line_idx))
                        raise e

    def process_from_file(self, sentence_path, pos_path, vocab) -> DataBundle:
        ds = DataSet()
        instances = []
        for idx, data in self.read_sentences(sentence_path):
            ins = {'raw_words': data}
            instances.append(ins)

        for idx, data in self.read_pos_tags(pos_path):
            instances[idx-1]['pos'] = data

        for ins in instances:
            ds.append(Instance(**ins))
        data_bundle = DataBundle(datasets = {'pred': ds})
        data_bundle = self.process(data_bundle, vocab)

        return data_bundle





def bmeso2bio(tags):
    new_tags = []
    for tag in tags:
        tag = tag.lower()
        if tag.startswith('m') or tag.startswith('e'):
            tag = 'i' + tag[1:]
        if tag.startswith('s'):
            tag = 'b' + tag[1:]
        new_tags.append(tag)
    return new_tags


def bmeso2bioes(tags):
    new_tags = []
    for tag in tags:
        lowered_tag = tag.lower()
        if lowered_tag.startswith('m'):
            tag = 'i' + tag[1:]
        new_tags.append(tag)
    return new_tags

def _indexize(data_bundle, input_field_names=Const.INPUT, target_field_names=Const.TARGET):
    if isinstance(input_field_names, str):
        input_field_names = [input_field_names]
    if isinstance(target_field_names, str):
        target_field_names = [target_field_names]
    for input_field_name in input_field_names:
        src_vocab = Vocabulary()
        src_vocab.from_dataset(*[ds for name, ds in data_bundle.iter_datasets() if 'train' in name],
                               field_name=input_field_name,
                               no_create_entry_dataset=[ds for name, ds in data_bundle.iter_datasets()
                                                        if ('train' not in name) and (ds.has_field(input_field_name))]
                               )
        src_vocab.index_dataset(*data_bundle.datasets.values(), field_name=input_field_name)
        data_bundle.set_vocab(src_vocab, input_field_name)
    
    if target_field_names:
        for target_field_name in target_field_names:
            tgt_vocab = Vocabulary(unknown=None, padding=None)
            tgt_vocab.from_dataset(*[ds for name, ds in data_bundle.iter_datasets() if 'train' in name],
                                   field_name=target_field_name,
                                   no_create_entry_dataset=[ds for name, ds in data_bundle.iter_datasets()
                                                            if ('train' not in name) and (ds.has_field(target_field_name))]
                                   )
            if len(tgt_vocab._no_create_word) > 0:
                warn_msg = f"There are {len(tgt_vocab._no_create_word)} `{target_field_name}` labels" \
                           f" in {[name for name in data_bundle.datasets.keys() if 'train' not in name]} " \
                           f"data set but not in train data set!.\n" \
                           f"These label(s) are {tgt_vocab._no_create_word}"
                #warnings.warn(warn_msg)
                logger.warning(warn_msg)
            tgt_vocab.index_dataset(*[ds for ds in data_bundle.datasets.values() if ds.has_field(target_field_name)], field_name=target_field_name)
            data_bundle.set_vocab(tgt_vocab, target_field_name)
    
    return data_bundle
