
import torch
from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import MarianTokenizer,MarianMTModel, MarianConfig

from utils.utils import MeanAggregator

class WMT14(BenchmarkSet):
    def __init__(self,batch_size=32) -> None:
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
      
        inputs = [ex['en'] for ex in data['translation']]
        targets = [ex['de'] for ex in data['translation']]
        model_inputs = self.tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=128, truncation=True, padding='max_length')
        model_inputs['labels'] = labels['input_ids']
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    def setup(self):
        reduced_dataset = self.dataset['train'].select(range(10000))
        self.dataset['train'] = reduced_dataset
        self.tokenized_datasets =self.dataset.map(self.preprocess, batched=True)
        self.tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
     
    def getDataLoader(self):
        train_set = DataLoader(self.tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        test_set = DataLoader(self.tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)
        val_set = DataLoader(self.tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        return (train_set,test_set,val_set)

    def getAssociatedModel(self):
        config = MarianConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=256,
            encoder_layers=6,
            encoder_ffn_dim=1024,
            encoder_attention_heads=4,
            decoder_layers=6,
            decoder_ffn_dim=1024,
            decoder_attention_heads=4,
            d_model=256,
            activation_function="relu",
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            init_std=0.02,
            classifier_dropout=0.1,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
    )
        self.model = MarianMTModel(config)
        print(f"Number of params: {sum(p.numel() for p in self.model.parameters())}")
        return  self.model

    def train(self, model, device, train_loader, optimizer, criterion,lr_scheduler):
        model.train()
        avg_loss = MeanAggregator()
        accuracy = MeanAggregator()

        translated = []
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
           # logits = outputs.logits

            # Flatten the logits and labels for computing the loss
            #loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            preds = torch.argmax(outputs.logits, dim=-1)
      
            mask = labels != self.tokenizer.pad_token_id
            correct = (preds[mask] == labels[mask]).sum().item()
          
            accuracy(correct/mask.sum().item())
            avg_loss(loss.item())
        
        return avg_loss.get(), accuracy.get()      
    @torch.no_grad()
    def test(self,model, device, test_loader, criterion):
        model.eval()
        accuracy = MeanAggregator()

        for batch in test_loader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            preds = torch.argmax(outputs.logits, dim=-1)
            mask = labels != self.tokenizer.pad_token_id
            correct = (preds[mask] == labels[mask]).sum().item()
            
            accuracy(correct/mask.sum().item())
        return accuracy.get()
    def translate(self, model,device, sentence: str) -> str:
        
        model.eval()  # Set the model to evaluation mode
        tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        with torch.no_grad():
            translated = model.generate(**(tokenizer([sentence], return_tensors="pt", padding=True).to(device)))

        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    def getAssociatedCriterion(self):
        pass

