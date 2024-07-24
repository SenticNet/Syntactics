from fastNLP import Vocabulary, DataSet, Instance, DataSetIter
from fastNLP import SequentialSampler

import torch
import torch.nn as nn
from flair.embeddings import FlairEmbeddings, WordEmbeddings, StackedEmbeddings
import flair
from flair.data import Sentence
from flair.tokenization import SpacyTokenizer



class FlairProcess():
    def __init__(self, dataset):
        self.embed = FlairEmbed()
        self.dataset = dataset
        self.count = 0
        
    def process(self):
        self.dataset.apply(self.toSentence,'raw_words')
        return self.dataset
        
    def toSentence(self,instance):
        self.count +=1
        if self.count%50==0:
            print(self.count)
        sentence = instance['raw_words']
        words = Sentence(sentence)
        # words.to(flair.device)
        words = self.embed(words)
        return words
            
                

class FlairEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        flair_embedding_forward = FlairEmbeddings('news-forward')
        flair_embedding_backward = FlairEmbeddings('news-backward')
        glove_embedding = WordEmbeddings('glove', force_cpu=False)
        self.embed = StackedEmbeddings([flair_embedding_forward, flair_embedding_backward,
                      glove_embedding])
                       
    def forward(self, sentence):
        self.embed.embed(sentence)
        names = self.embed.get_names()
        all_embs = [emb.to('cpu') for token in sentence for emb in token.get_each_embedding(names)]
        sentence = torch.cat(all_embs).view([-1,self.embed.embedding_length])
        # sentence.to('cpu')
        return sentence

