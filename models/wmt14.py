
from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import MarianTokenizer,MarianMTModel, MarianConfig

class WMT14(BenchmarkSet):
    def __init__(self) -> None:
        super().__init__()
        self.conf = getConfig()
      # Load the WMT14 English-German dataset
        self.dataset = load_dataset("wmt14", "de-en")

        self.tokenizer =   MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        self.setup()

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
        self.tokenized_datasets =self.dataset.map(self.preprocess, batched=True)
        self.tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
    def getDataLoader(self):
       pass

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
        pass

