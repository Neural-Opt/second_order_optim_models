
from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import MarianTokenizer,MarianMTModel, MarianConfig

class WMT14(BenchmarkSet):
    def __init__(self) -> None:
        super().__init__()
        self.conf = getConfig()
      # Load the WMT14 English-German dataset
        self.dataset = datasets.load_dataset("wmt14", "de-en")
        self.tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')


    def log(self):
        pass

    def preprocess(self,data):
        inputs = [ex['en'] for ex in data['translation']]
        targets = [ex['de'] for ex in data['translation']]
        model_inputs = self.tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=128, truncation=True, padding='max_length')
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    def setup(self):
       # Apply preprocessing
        self.tokenized_datasets =self.dataset.map(self.preprocess, batched=True)
        self.tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
    def getDataLoader(self):
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {x: datasets.ImageFolder(root=f"{self.dataset_path}/{x}", transform=data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        return dataloaders['train'], dataloaders['val']

    def getAssociatedModel(self):
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
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
    )
        model = MarianMTModel(config)
        return model

    def getAssociatedCriterion(self):
        return nn.CrossEntropyLoss()

