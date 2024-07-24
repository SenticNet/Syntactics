import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--ID', default='0', help='run ID')
parser.add_argument('--gpu', default='1,6', type=str, help='gpu device numbers')
parser.add_argument('--load_trained', type=bool, default=True, help='load trained model or not')
parser.add_argument('--logger_path', type=str, default='log.txt', help='logger path')

parser.add_argument('--epoch', type=int, default=50,help='random seed')
parser.add_argument('--batch_size', type=int, default=20,help='batch_size')
parser.add_argument('--num_workers', type=int, default=4,help='num_of_workers')
parser.add_argument('--lr', type=float, default=1e-4,help='learning rate')
parser.add_argument('--p_paths', type=str, default="p_dataset", help='task p dataset file path')
parser.add_argument('--c_paths', type=str, default="c_dataset", help='task c dataset file path')
parser.add_argument('--cl_size', type=int, default=4,help='curriculum learning size')
parser.add_argument('--num_layers', type=int, default=2,help='number of layers')

parser.add_argument('--print_every', type=int, default=100,help='how many iter for averaging results')
parser.add_argument('--seed', type=int, default=666,help='random seed')
# parser.add_argument('--sent_max_length', type=int, default=60,help='sent_max_length')

args = parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch
import random
from fastNLP import DataSet, Vocabulary, Trainer, Tester
from fastNLP import GradientClipCallback, WarmupCallback
from fastNLP import SpanFPreRecMetric, BucketSampler, AccuracyMetric, ClassifyFPreRecMetric
from modules.multitaskTagger import MultitaskTagger
from modules.pipe import MultitaskFlairPipe
from modules.multitask_trainer import Multitask_Trainer
from modules.multitask_tester import Multitask_Tester
from flair.embeddings import FlairEmbeddings, WordEmbeddings, CharacterEmbeddings, StackedEmbeddings

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main(args):
    p_paths = {'test': args.p_paths + "/test.txt",
        'dev': args.p_paths + "/dev.txt",
        'train': args.p_paths + "/train.txt"}
    c_paths = {'test': args.c_paths + "/test.txt",
        'dev': args.c_paths + "/dev.txt",
        'train': args.c_paths + "/train.txt"}
    data_bundle = MultitaskFlairPipe().process_from_file([c_paths, p_paths])

    p_vocab = data_bundle.get_vocab('p')
    c_vocab = data_bundle.get_vocab('c')
    flair_embedding_forward = FlairEmbeddings('news-forward')
    flair_embedding_backward = FlairEmbeddings('news-backward')
    glove_embedding = WordEmbeddings('glove', force_cpu=False)
    char_embedding = CharacterEmbeddings()
    embed = StackedEmbeddings([flair_embedding_forward, flair_embedding_backward, glove_embedding, char_embedding])

    model = MultitaskTagger(embed, len(p_vocab), len(c_vocab), cl_size=args.cl_size, num_layers=args.num_layers, c_vocab=c_vocab)
    if torch.cuda.is_available():
        model = model.to(device)

    callbacks = []
    clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
    warmup_callback = WarmupCallback(warmup=0.1, schedule='constant')
    callbacks.extend([clip_callback, warmup_callback])

    acc_metric = AccuracyMetric()
    p_metric = ClassifyFPreRecMetric(tag_vocab=p_vocab)
    # c_metric = ClassifyFPreRecMetric(tag_vocab=c_vocab)
    c_metric = SpanFPreRecMetric(tag_vocab=c_vocab)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    multi_trainer = Multitask_Trainer(data_bundle.get_dataset('train'), model, optimizer, 
                add_data=data_bundle.get_dataset('p_train'), batch_size=args.batch_size, 
                sampler=BucketSampler(), num_workers=args.num_workers, n_epochs=args.epoch, 
                dev_data=data_bundle.get_dataset('p_dev'), 
                test_data=[data_bundle.get_dataset('p_test'), data_bundle.get_dataset('test')],
                metrics=[acc_metric, p_metric], c_metrics=[c_metric, acc_metric], callbacks=callbacks, 
                device=device, test_use_tqdm=False, use_tqdm=True, print_every=args.print_every, 
                save_path=None, logger_path=args.logger_path)
    multi_trainer.train(load_best_model=args.load_trained)
    
    p_tester = Tester(data_bundle.get_dataset('p_test'), model=model, metrics=[acc_metric, p_metric],
                        batch_size=args.batch_size, verbose=0)
    tester = Multitask_Tester(data_bundle.get_dataset('test'), model=model, metrics=[acc_metric, c_metric],
                        batch_size=args.batch_size, verbose=0)
    p_tester.test()
    tester.test()

if __name__ == "__main__":
    
    
    main(args)