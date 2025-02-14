import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

NEED_TOKEN = 'xNeed'
EFFECT_TOKEN = 'xEffect'
INTENT_TOKEN = 'xIntent'
REACT_TOKEN = 'xReact'



def main():

    dataset = load_dataset('json', data_files="./scripts/comet/data.jsonl", split='train')
    raw_datasets = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

    tokenizer = AutoTokenizer.from_pretrained('nlp-waseda/gpt2-small-japanese')
    print(len(tokenizer))
    special_tokens_dict = {
        'additional_special_tokens': [
            NEED_TOKEN,
            EFFECT_TOKEN,
            INTENT_TOKEN,
            REACT_TOKEN,
        ]
    }
    print(len(tokenizer))
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(len(tokenizer), tokenizer.vocab_size)

    def preprocess_function(examples):
        outputs = []

        for head_text, inf_type_dict in zip(examples['event'], examples['inference']):
            for inf_type, inf_dir_dict in inf_type_dict.items():
                if inf_dir_dict is None:
                    continue

                for inf_dir, tail_list in inf_dir_dict.items():
                    if tail_list is None:
                        continue

                    if inf_type == 'event':
                        if inf_dir == 'before':
                            rel_token = NEED_TOKEN
                        else:
                            rel_token = EFFECT_TOKEN
                    else:
                        if inf_dir == 'before':
                            rel_token = INTENT_TOKEN
                        else:
                            rel_token = REACT_TOKEN

                    for tail_text in tail_list:
                        output = head_text + rel_token + tail_text + tokenizer.eos_token
                        outputs.append(output)

        return {'data': outputs}
    
    preprocessed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    tokenized_datasets = preprocessed_datasets.map(
        lambda examples: tokenizer(
            examples['data'],
            truncation=True,
            max_length=128,
        ),
        batched=True,
        remove_columns=preprocessed_datasets['train'].column_names,
    )

    #model = AutoModelForCausalLM.from_pretrained('nlp-waseda/gpt2-small-japanese')
    #model.resize_token_embeddings(len(tokenizer))

    args = TrainingArguments(
        output_dir="./temp",
        evaluation_strategy='epoch',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        logging_strategy='epoch',
        save_strategy='no',
    )

    class DataCollatorForComet(DataCollatorForLanguageModeling):

        def torch_call(self, examples):

            batch = super().torch_call(examples)            
            labels = batch['labels']

            rel_mask = labels >= self.tokenizer.vocab_size
            tail_mask = (rel_mask.cumsum(dim=-1) - rel_mask.to(int)).to(bool)
            labels[~tail_mask] = -100

            batch['labels'] = labels
            return batch
        

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForComet(
        tokenizer=tokenizer,
        mlm=False,
    )

    print(data_collator)
    
    dummy_texts = [
        "イベントA " + NEED_TOKEN + " 結果B " + tokenizer.eos_token,
        "イベントC " + REACT_TOKEN + " 結果D " + tokenizer.eos_token,
    ]
    encoded = tokenizer(dummy_texts, truncation=True, max_length=32, padding='max_length', return_tensors='pt')
    print(encoded)
    dummy_examples = []
    for i in range(encoded['input_ids'].size(0)):
        dummy_examples.append({'input_ids': encoded['input_ids'][i]})
    batch = data_collator(dummy_examples)
    print(batch)



    """trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model()"""


if __name__ == '__main__':
    main()