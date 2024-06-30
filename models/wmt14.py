
import torch
from benchmark.benchmark import Benchmark
from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import MarianTokenizer,MarianMTModel, MarianConfig
#from torchtext.data.metrics import bleu_score

from utils.utils import MeanAggregator

class WMT14(BenchmarkSet):
    def __init__(self,batch_size=1) -> None:
        super().__init__()
        self.conf = getConfig()
      # Load the WMT14 English-German dataset
        self.dataset = load_dataset("wmt14", "de-en")

        self.tokenizer =   MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
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
        return self.tokenizer(inputs, text_target=targets, max_length=128, truncation=True, padding='max_length')
    def setup(self):
        reduced_dataset = self.dataset['train'].select(range(1))
     
        self.dataset['train'] = reduced_dataset
        self.tokenized_datasets = self.dataset.map(self.preprocess, batched=True)
        self.tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    def getDataLoader(self):
        train_set = DataLoader(self.tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        test_set = DataLoader(self.tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)
        val_set = DataLoader(self.tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        return (train_set,test_set,val_set)

    def getAssociatedModel(self,rank):
        config = MarianConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=512,
            encoder_layers=6,
            encoder_ffn_dim=2048,
            encoder_attention_heads=8,
            decoder_layers=6,
            decoder_ffn_dim=2048,
            decoder_attention_heads=8,
            d_model=512,
            activation_function="relu",
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            init_std=0.02,
            classifier_dropout=0.1,
            num_beams=4,
            length_penalty=0.6,
            early_stopping=True,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
    )
        self.model = MarianMTModel(config)
        self.model.to(rank)
        print(f"Number of params: {sum(p.numel() for p in self.model.parameters())}")
        ddp_model = torch.nn.DataParallel(self.model)

        return  ddp_model

    def train(self, model, device, train_loader, optimizer, criterion, create_graph):
        model.train()
        benchmark = Benchmark.getInstance(None)

        avg_loss = MeanAggregator()
        accuracy = MeanAggregator()

        for batch in train_loader:
        
            benchmark.measureGPUMemUsageStart(rank=device)

            benchmark.stepStart()
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
        
            loss = outputs.loss.mean()
            loss.backward(create_graph=create_graph)
            optimizer.step()
            preds = torch.argmax(outputs.logits, dim=-1)
            mask = batch['labels'] != self.tokenizer.pad_token_id
            correct = (preds[mask] == batch['labels'][mask]).sum().item()
          
            accuracy(correct/mask.sum().item())
            avg_loss(loss.item())
            benchmark.stepEnd()
            benchmark.measureGPUMemUsageEnd(rank=device)

        benchmark.addTrainAcc(accuracy.get())
        benchmark.addTrainLoss(avg_loss.get())
        benchmark.flush()
        return avg_loss.get(), accuracy.get()      
    @torch.no_grad()
    def test(self,model, device, test_loader, criterion):
        model.eval()
        benchmark = Benchmark.getInstance(None)
        accuracy = MeanAggregator()
        bleu = MeanAggregator()
        sentence = "Ich bitte Sie, sich zu einer Schweigeminute zu erheben."
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            preds = torch.argmax(outputs.logits, dim=-1)
            mask = batch['labels']!= self.tokenizer.pad_token_id
            correct = (preds[mask] == batch['labels'][mask]).sum().item()
            decoded_references = [self.tokenizer.decode(labels, skip_special_tokens=False) for labels in batch['labels']]
            decoded_predictions = [self.tokenizer.decode(preds_batch, skip_special_tokens=False) for preds_batch in preds]
            
            bleu(bleu_score(decoded_references, decoded_predictions, max_n=4))
            accuracy(correct/mask.sum().item())
           
        benchmark.addTestAcc(accuracy.get())
        print("DE: "+sentence + "EN: "+self.translate(model.module,device,sentence))
        return accuracy.get()
    def translate(self, model,device, sentence: str) -> str:
        
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            inputs = self.tokenizer([sentence], return_tensors="pt", padding=True).to(device)
            translated = model.generate(**inputs)
        txt = self.tokenizer.batch_decode(translated, skip_special_tokens=False)[0]
        references = [['the', 'cat', 'is', 'on', 'the', 'mat']]
        candidates = [['the', 'cat', 'sat', 'on', 'the', 'mat']]

        # Compute BLEU score
        score = bleu_score(candidates, references)

        print(txt)
        return txt

    def getAssociatedCriterion(self):
        pass

