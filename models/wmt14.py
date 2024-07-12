
import torch
from benchmark.benchmark import Benchmark
from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import MarianTokenizer,MarianMTModel, MarianConfig
from sacrebleu import corpus_bleu
import numpy as np
from utils.utils import MeanAggregator

class WMT14(BenchmarkSet):
    def __init__(self,batch_size=256) -> None:
        super().__init__()
        self.conf = getConfig()
      # Load the WMT14 English-German dataset
        self.dataset = load_dataset("wmt14", "de-en")

        self.tokenizer =  MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        print(f"Vocab size {self.tokenizer.vocab_size}")
        self.batch_size = batch_size
        self.setup()

    def log(self):
        pass
    def decode_tensor(self,tensor):
        decoded_sentences = []
        for sequence in tensor:
            # Decode the sequence and skip special tokens
            decoded_sentence = self.tokenizer.decode(sequence, skip_special_tokens=True)
            decoded_sentences.append([decoded_sentence])
        return decoded_sentences

    def preprocess(self,data):
        inputs = [ex['de'] for ex in data['translation']]
        targets = [ex['en'] for ex in data['translation']]
        return self.tokenizer(inputs, text_target=targets, max_length=32, truncation=True, padding='max_length')
    def setup(self):

       # print(len(self.dataset['train']))
        self.dataset['train'] =  self.dataset['train'].select(range(5))
        self.dataset['test'] =  self.dataset['test'].select(range(5))

        self.tokenized_datasets = self.dataset.map(self.preprocess, batched=True,load_from_cache_file=False)
        self.tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    def getDataLoader(self):
        train_set = DataLoader(self.tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        test_set = DataLoader(self.tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)
        val_set = DataLoader(self.tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        return (train_set,test_set,val_set)

    def getAssociatedModel(self,rank):
        config = MarianConfig(
           vocab_size=self.tokenizer.vocab_size,
            encoder_layers=3,
            encoder_ffn_dim=512,
            encoder_attention_heads=8,
            decoder_layers=3,
            decoder_ffn_dim=512,
            decoder_attention_heads=8,
            d_model=512,     
    )
        print(self.tokenizer.pad_token_id)
        self.model = MarianMTModel(config)
        self.model.to(rank)
        print(f"Number of params: {sum(p.numel() for p in self.model.parameters())}")
        ddp_model = torch.nn.DataParallel(self.model)

        return  ddp_model
    def loss_function(self,logits, labels):
        logits = logits.view(-1, logits.size(-1))  # Shape: [batch_size * sequence_length, num_classes]
        labels = labels.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(logits, labels)
        return loss
    def train(self, model, device, train_loader, optimizer, criterion, create_graph):
       
        benchmark = Benchmark.getInstance(None)


        avg_loss = MeanAggregator()
        accuracy = MeanAggregator()

        for batch in train_loader:
        
            benchmark.measureGPUMemUsageStart(rank=device)

            benchmark.stepStart()
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
              input_ids=batch["input_ids"],
              attention_mask=batch["attention_mask"],
              labels = torch.where(batch['labels'] == self.tokenizer.pad_token_id, torch.tensor(-100), batch['labels'])
              )
         
            loss =outputs.loss #self.loss_function(outputs.logits,batch['labels'])
           
            loss.backward(create_graph=create_graph)
            optimizer.step()
            preds = torch.argmax(outputs.logits, dim=-1)
         
            mask = batch['labels'] != self.tokenizer.pad_token_id

            correct = (preds[mask] == batch['labels'][mask]).sum().item()
          
            accuracy(correct/(mask.sum().item()))
            avg_loss(loss.item())
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

        accuracy = MeanAggregator()

        decoded_references=[]
        decoded_predictions=[]
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            preds = torch.argmax(outputs.logits, dim=-1)
            mask = batch['labels']!= self.tokenizer.pad_token_id

            correct = (preds[mask] == batch['labels'][mask]).sum().item()
            decoded_references = decoded_references + [[self.tokenizer.decode(labels.tolist(), skip_special_tokens=True)] for labels in batch['labels']]
            decoded_predictions = decoded_predictions + [self.tokenizer.decode(preds_batch.tolist(), skip_special_tokens=True) for preds_batch in preds]            
            accuracy(correct/mask.sum().item())

        benchmark.add("acc_test",accuracy.get())
        sacre_bleu = corpus_bleu(decoded_predictions, decoded_references,use_effective_order=True)
       
        print(f"Bleu: {sacre_bleu.score} ")


      #  self.translate(model,device,"I am going to buy a car!")
        benchmark.add("bleu",sacre_bleu.score)

        return accuracy.get()
    def translate(self, model,device, sentence: str) -> str:
        
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            inputs = self.tokenizer([sentence], return_tensors="pt", padding=True).to(device)
            translated = model.module.generate(**inputs)
        txt = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
       # print(txt)
        return txt

    def getAssociatedCriterion(self):
        pass

