
import torch
from benchmark.benchmark import Benchmark
from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import MarianTokenizer,MarianMTModel, MarianConfig, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import math
import evaluate
from utils.utils import MeanAggregator

class InverseSquareRootLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, init_lr, min_lr=1e-9, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.min_lr = min_lr
        super(InverseSquareRootLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.init_lr * (step / self.warmup_steps)
        else:
            # Inverse square root decay
            lr = self.init_lr * math.sqrt(self.warmup_steps / step)

        # Ensure learning rate doesn't go below minimum
        lr = max(lr, self.min_lr)

        return [lr for _ in self.base_lrs]

class WMT14(BenchmarkSet):
    def __init__(self,batch_size=256,epochs = 164) -> None:
        super().__init__()
        self.conf = getConfig()
        self.metric = evaluate.load("sacrebleu")

        self.dataset = load_dataset("wmt14", "de-en")
        self.tokenizer =  MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
       
        self.batch_size = batch_size
        self.lr_scheduler =None
        self.scaler = GradScaler()


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
        inputs = [ex['en'] for ex in data['translation']]
        targets = [ex['de'] for ex in data['translation']]
        return self.tokenizer(inputs, text_target=targets, max_length=128, truncation=True, padding='max_length')
    def getLRScheduler(self,optim):
        if self.lr_scheduler == None:
            self.lr_scheduler =  InverseSquareRootLR(
                                        optim,
                                        warmup_steps=40,
                                        min_lr=1e-9,
                                        init_lr=optim.param_groups[0]['lr']
)
        return self.lr_scheduler
    def setup(self):

       # print(len(self.dataset['train']))
        self.dataset['train'] =  self.dataset['train'].select(range(10000))
        self.dataset['test'] =  self.dataset['test'].select(range(1000))

       
        self.tokenized_datasets = self.dataset.map(self.preprocess, batched=True,load_from_cache_file=False)
        self.tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
   
    def getDataLoader(self):
        train_set = DataLoader(self.tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        test_set = DataLoader(self.tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)
        val_set = DataLoader(self.tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        return (train_set,test_set,len(train_set))

    def getAssociatedModel(self,rank):
        config = MarianConfig(
           vocab_size=self.tokenizer.vocab_size,
            encoder_layers=6,
            encoder_ffn_dim=2048,
            encoder_attention_heads=8,
            decoder_layers=6,
            decoder_ffn_dim=2048,
            decoder_attention_heads=8,
            d_model=512,     
    )
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
    def train(self, model, device, train_loader, optimizer, criterion, create_graph,lr_scheduler):
        model.train()
        lr_scheduler = self.getLRScheduler(optimizer)
        benchmark = Benchmark.getInstance(None)


        avg_loss = MeanAggregator()
        accuracy = MeanAggregator()

        for batch in train_loader:
            benchmark.measureGPUMemUsageStart(rank=device)

            benchmark.stepStart()
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = torch.where(batch['labels'] == self.tokenizer.pad_token_id, torch.tensor(-100), batch['labels'])

            with autocast():
                outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels
                )
                loss = outputs.loss.mean()

      #self.loss_function(outputs.logits,batch['labels'])
            self.scaler.scale(loss).backward(create_graph=create_graph)
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)

            # Update scaler
            self.scaler.update()
            lr_scheduler.step()
            preds = torch.argmax(outputs.logits, dim=-1)
         
            mask = batch['labels'] != self.tokenizer.pad_token_id

            correct = (preds[mask] == batch['labels'][mask]).sum().item()
          
            accuracy(correct/(mask.sum().item()))
            avg_loss(loss.item())
            benchmark.stepEnd()
            benchmark.measureGPUMemUsageEnd(rank=device)
            benchmark.add("train_loss",avg_loss.get())


        benchmark.add("acc_train",accuracy.get())
          
        benchmark.flush()
        return avg_loss.get(), accuracy.get()      
    @torch.no_grad()
    def test(self,model, device, test_loader, criterion):
        model.eval()
        print("TEST NOW")
        benchmark = Benchmark.getInstance(None)

        decoded_references=[]
        decoded_predictions=[]
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model.module.generate(
              input_ids=batch["input_ids"],
              attention_mask=batch["attention_mask"])
             
            decoded_references = decoded_references + [[self.tokenizer.decode(labels.tolist(), skip_special_tokens=True)] for labels in batch['labels']]
            decoded_predictions = decoded_predictions + [self.tokenizer.decode(preds_batch.tolist(), skip_special_tokens=True) for preds_batch in preds]            
           # accuracy(correct/(mask.sum().item()))

        result = self.metric.compute(predictions=decoded_predictions, references=decoded_references, use_effective_order=True)


        for i in range(5):
            print(f"\nREF: {decoded_references[i][0]}")
            print(f"PRED: {decoded_predictions[i]}")
           # print(f"BLEU: {corpus_bleu([decoded_references[i]],[decoded_predictions[i]],use_effective_order=True).score}")

        
        benchmark.add("bleu",result["score"])

        return result["score"]
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

