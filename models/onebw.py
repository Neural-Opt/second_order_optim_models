from benchmark.benchmark import Benchmark
from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from models.lstm.LSTM import LSTM
from models.resnet.resnet110 import ResNet110
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader,DistributedSampler, random_split
from torch.nn.parallel import DistributedDataParallel as DP
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch.nn as nn
import torch
import numpy as np
import random
import pickle
import os
import re
import string

from utils.utils import MeanAggregator
from collections import defaultdict

class Vocabulary:
    def __init__(self,threshold=3):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1,"<SOS>": 2,"<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>",2:"<SOS>",3:"<EOS>"}
        self.idx = 4
        self.word_counter = defaultdict(int)
        self.threshold = threshold

    def add_word(self, word):
        self.word_counter[word] += 1
        if self.word_counter[word] >= self.threshold:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)
    
class CachedDataset(Dataset):
    def __init__(self, cached_file):
        print("Loading dataset")
        with open(cached_file, 'rb') as f:
            self.data = pickle.load(f)
        print("Finished Loading dataset")
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])
    
class OneBW(BenchmarkSet):
    def __init__(self,batch_size=1) -> None:
        super().__init__()
        self.mode = "_TEST"
        self.conf = getConfig()
        self.batch_size = batch_size
        self.dataset = load_dataset("lm1b")

        self.pad_length = 64
        self.cache_dir = "./data/lm1b_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if self.mode == "_TEST":
            self.dataset["train"] = self.dataset["train"].select(range(5000))
            self.dataset["test"] = self.dataset["test"].select(range(5000))

        self.vocab = self.build_and_cache_vocab()
        print(f"Calc Vocab size")

        self.vocab_size = len(self.vocab.word2idx.keys())
        print(f"Vocab size: { self.vocab_size}")
        self.cache_dataset()
    def process_text(self,text):
        text = text.lower()    
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)    
        text = re.sub(r'\d+', '', text)
        return text
    
    def build_and_cache_vocab(self):


        vocab_file = os.path.join(self.cache_dir, f'vocab{self.mode}.pkl')
        print(vocab_file)
        if os.path.exists(vocab_file):
            print("Load vocabulary from cache...")
            with open(vocab_file, 'rb') as f:
                vocab = pickle.load(f)
        else:
            print("Building vocabulary...")
            vocab = Vocabulary()
            train_texts = self.dataset['train']['text']
            for text in tqdm(train_texts, desc="Processing texts"):
                text = self.process_text(text)
                for word in text.split():
                    vocab.add_word(word)
            print(f"Dump Vocab")
            with open(vocab_file, 'wb') as f:
                pickle.dump(vocab, f)
        return vocab
    
    def cache_dataset(self):
        for split in ['train', 'test']:
            cache_file = os.path.join(self.cache_dir, f"{split}_cached{self.mode}.pkl")
            if not os.path.exists(cache_file):
                print(f"Caching {split} split...")
                texts = self.dataset[split]['text']
                tokenized_data = []
                for text in tqdm(texts, desc=f"Caching {split} split..."):
                    text = self.process_text(text)
                    tokenized_text = [self.vocab.word2idx.get("<SOS>")] + [self.vocab.word2idx.get(word, self.vocab.word2idx["<UNK>"]) for word in text.split()]
                    tokenized_text = (tokenized_text[:self.pad_length-1] if len(tokenized_text) > self.pad_length-1 else tokenized_text)  + [self.vocab.word2idx.get("<EOS>")]
                    padded_text = tokenized_text + [self.vocab.word2idx["<PAD>"]] * (self.pad_length - len(tokenized_text))

                    tokenized_data.append(padded_text)
                with open(cache_file, 'wb') as f:
                    pickle.dump(tokenized_data, f)
            else:
                 print(f"Using cache for {split} split...")

    def log(self):
        pass
    def setup(self,):
        pass
    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    def collate_fn(self, batch):
        batch = torch.stack(batch, dim=0)
        return batch
    def getDataLoader(self,):   
        g = torch.Generator()
        g.manual_seed(404)
        train_dataset = CachedDataset(os.path.join(self.cache_dir, f'train_cached{self.mode}.pkl'))
        test_dataset = CachedDataset(os.path.join(self.cache_dir, f'test_cached{self.mode}.pkl'))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, worker_init_fn=self.seed_worker, generator=g, collate_fn=self.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, worker_init_fn=self.seed_worker, generator=g, collate_fn=self.collate_fn)
        
 
        return (train_loader ,test_loader , None)
    def getAssociatedModel(self,rank):
        model = LSTM(vocab_size=self.vocab_size,
             embed_size=1,
             hidden_size=1,
             num_layers=2)
        print(f"Number of params: {sum(p.numel() for p in model.parameters())}")

        model = model.to(rank)
        ddp_model = torch.nn.DataParallel(model)#, device_ids=[rank])

        return model
    def getAssociatedCriterion(self):
        return nn.CrossEntropyLoss()
    def train(self, model, device, train_loader, optimizer, criterion,create_graph):
        model.train()

        print("1BW")
        benchmark = Benchmark.getInstance(None)
        accuracy = MeanAggregator(measure=lambda *args:(args[0].eq(args[1]).sum().item() / args[1].size(0)))
        avg_loss = MeanAggregator()
        for inputs in train_loader:
      
            #print(inputs.shape,len(train_loader))
           # print([self.vocab.idx2word[w.item()] for w in inputs[67]])
           
            benchmark.measureGPUMemUsageStart(rank=device)
            benchmark.stepStart()
            hidden = model.init_hidden(self.batch_size,device)

            inputs = inputs.to(device)
            optimizer.zero_grad()
            lengths = (inputs != self.vocab.word2idx["<PAD>"]).sum(dim=1).long().to(device)
           # print(lengths.shape,(inputs != self.vocab.word2idx["<PAD>"]).sum(dim=1).shape)
            #print((inputs != self.vocab.word2idx["<PAD>"])[67])
            print(lengths.shape)#torch.Size([128])
            print(lengths.get_device()) #prints -1
            print(lengths.dtype) #prints -1
            

            outputs,hidden = model(inputs,hidden)
            raise Exception()

            loss = criterion(outputs, targets)
            loss.backward(create_graph=create_graph) 
            optimizer.step()
            _, predicted = outputs.max(1)

            avg_loss(loss.item())
            accuracy(predicted,targets)
            benchmark.stepEnd()
            benchmark.measureGPUMemUsageEnd(rank=device)

        benchmark.add("acc_train",accuracy.get())
        benchmark.add("train_loss",avg_loss.get())
        benchmark.flush()
       
        return avg_loss.get(), accuracy.get()
    @torch.no_grad()
    def test(self,model, device, test_loader, criterion):
        model.eval()
        benchmark = Benchmark.getInstance(None)

        accuracy = MeanAggregator(measure=lambda *args:(args[0].eq(args[1]).sum().item() / args[1].size(0)))
        avg_loss = MeanAggregator()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                _, predicted = outputs.max(1)
                avg_loss(loss.item())
                accuracy(predicted,targets)

            benchmark.add("acc_test",accuracy.get())
            benchmark.add("test_loss",avg_loss.get())
        return accuracy.get()
        