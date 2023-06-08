# %%
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json", 
                                    bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>")

# %%
from transformers import BartForConditionalGeneration, BartConfig
import torch
import json

config_facebook_bart_base = json.load(open("config_facebook_bart_base.json", "r"))

del config_facebook_bart_base['_name_or_path']
del config_facebook_bart_base['task_specific_params']
del config_facebook_bart_base['transformers_version']
config_facebook_bart_base['vocab_size'] = tokenizer.vocab_size
config_facebook_bart_base['bos_token_id'] = tokenizer.bos_token_id
config_facebook_bart_base['forced_bos_token_id'] = tokenizer.bos_token_id
config_facebook_bart_base['eos_token_id'] = tokenizer.eos_token_id
config_facebook_bart_base['forced_eos_token_id'] = tokenizer.eos_token_id
config_facebook_bart_base['pad_token_id'] = tokenizer.pad_token_id
config_facebook_bart_base['decoder_start_token_id'] = tokenizer.eos_token_id
config_facebook_bart_base['torch_dtype'] = torch.float16

config = BartConfig(**config_facebook_bart_base)

model = BartForConditionalGeneration(config)

# %%
# This function is copied from modeling_bart.py
def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

# %%
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['tokenized_kana_text'], 
                                                  pad_to_max_length=True, max_length=512, return_tensors="pt")
    target_encodings = tokenizer.batch_encode_plus(example_batch['plain_text'], 
                                                   pad_to_max_length=True, max_length=512, return_tensors="pt")

    labels = target_encodings['input_ids']
    decoder_input_ids = shift_tokens_right(labels, tokenizer.pad_token_id)
    labels[labels[:, :] == tokenizer.pad_token_id] = -100

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': decoder_input_ids,
        'labels': labels
    }

    return encodings


# %%
from datasets import Dataset, DatasetDict
import os

if not os.path.exists("dataset_all"):
    dataset_all = Dataset.from_json("all.json")
    dataset_sub = dataset_all
    # dataset_sub = Dataset.from_dict(dataset_all[:500])
    num_test_samples = 1000
    test_size = num_test_samples / len(dataset_sub)
    dataset_dict = Dataset.train_test_split(dataset_sub, test_size=test_size)
    dataset_dict = dataset_dict.map(convert_to_features, batched=True)
    dataset_dict.save_to_disk("dataset_all")
else:
    dataset_dict = DatasetDict.load_from_disk("dataset_all")

columns = ['input_ids', 'labels', 'decoder_input_ids','attention_mask',] 
dataset_dict.set_format(type='torch', columns=columns)

# %%
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./models/kana-kanji',
    num_train_epochs=50,
    per_device_train_batch_size=20, 
    per_device_eval_batch_size=1,   
    warmup_steps=500,               
    weight_decay=0.01,              
    logging_dir='./logs',
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,                       
    args=training_args,                  
    train_dataset=dataset_dict['train'],        
    eval_dataset=dataset_dict['test']   
)

# %%
trainer.train()

# %%
model.save_pretrained("./models/kana-kanji_230604")

# # %%
# input_text = dataset_all[234]['tokenized_kana_text']
# print(input_text)

# # %%
# input_ids=tokenizer.encode(input_text, return_tensors="pt").to("cuda")
# print(input_ids)

# # %%
# output_ids = model.generate(input_ids, max_length=1024, num_beams=5, early_stopping=True)
# print(output_ids)

# # %%
# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print(output_text)


