from sacrebleu import corpus_bleu
import nltk
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import MarianTokenizer, MarianMTModel, MarianConfig, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from transformers.utils.logging import set_verbosity
import logging
import transformers

class CustomCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metric = evaluate.load("sacrebleu")

    def on_epoch_end(self, args, state, control, **kwargs):
        pass

    def compute_bleu(self, preds, labels):
        pass
    def log(self, logs):
        """Override the default log method to suppress unwanted logs."""
        if 'eval_loss' in logs or 'loss' in logs:
            return
        super().log(logs)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load and prepare the dataset
wmt14 = load_dataset("wmt14", "de-en", split="train").shuffle(seed=42)
train_size = 250000
test_size = 1000

# Select 10,000 samples for training and 1,000 samples for testing
train_dataset = wmt14.select(range(train_size))
test_dataset = wmt14.select(range(train_size, train_size + test_size))

src_lang = "de"
tgt_lang = "en"

# Initialize tokenizer and model from configuration (not pretrained)
tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}')
config = MarianConfig(
   vocab_size=tokenizer.vocab_size,
            encoder_layers=6,
            encoder_ffn_dim=2048,
            encoder_attention_heads=8,
            decoder_layers=6,
            decoder_ffn_dim=2048,
            decoder_attention_heads=8,
            d_model=512,   
)
model = MarianMTModel(config)

def preprocess_function(examples):
    #print(examples)
    inputs = [ex[src_lang] for ex in examples['translation']]
    targets = [ex[tgt_lang] for ex in examples['translation']]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Setup evaluation
nltk.download("punkt", quiet=True)
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_labels = [[label] for label in decoded_labels]
    

    bleu = corpus_bleu(decoded_preds,decoded_labels,use_effective_order=True)

    print("---------------------------")
    print(f"BLEU: {bleu.score}")
    print("---------------------------")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)


    return result

# Initialize data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Setup training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    logging_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=0,
    num_train_epochs=100,
    fp16=True,
    predict_with_generate=True,
    logging_dir='./logs', 
)
print("TRAIN")
# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

)
#trainer.add_callback(CustomCallback(tokenizer))
# Start training
trainer.train()




