import json, sys, datasets, argparse, os, wandb

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, Trainer

sys.path.append("./scripts/utils")
from load_data import load_gold, load_silver


ADV_TOKEN = "xEffect"



def load_dataset(settings_data, patent_domain, tokenizer):

    gold_data = load_gold(settings_data, "patent", patent_domain)
    silver_data = load_silver(settings_data, "patent", patent_domain)

    datasets_dict = datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"data": [d[0] + ADV_TOKEN + d[1] for d in silver_data[:int(len(silver_data) * 0.9)]]}),
        "validation": datasets.Dataset.from_dict({"data": [d[0] + ADV_TOKEN + d[1] for d in silver_data[int(len(silver_data) * 0.9):]]}),
        "test": datasets.Dataset.from_dict({"data": [d[0] + ADV_TOKEN + d[1] for d in gold_data]})
    })

    tokenized_datasets = datasets_dict.map(
        lambda examples: tokenizer(
            examples["data"],
            truncation = True,
            max_length = 256,
        ),
        batched = True,
        remove_columns = datasets_dict["train"].column_names
    )

    return tokenized_datasets



class DataCollatorForComet(DataCollatorForLanguageModeling):

    def torch_call(self, examples):

        batch = super().torch_call(examples)            
        labels = batch['labels']

        rel_mask = labels >= self.tokenizer.vocab_size
        tail_mask = (rel_mask.cumsum(dim=-1) - rel_mask.to(int)).to(bool)
        labels[~tail_mask] = -100

        batch['labels'] = labels

        return batch



def main(settings_comet, settings_data, patent_domain):

    model = AutoModelForCausalLM.from_pretrained(settings_comet["train_parameters"]["model_name"]).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(settings_comet["train_parameters"]["model_name"])
    datasets_dict = load_dataset(settings_data, patent_domain, tokenizer)

    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer), "not added properly"
    assert len(tokenizer.encode(ADV_TOKEN, add_special_tokens=False)) == 1, "not added properly"

    args = TrainingArguments(
        do_train = True,
        do_eval = True,
        eval_strategy = "steps",
        eval_steps = 1000,
        logging_steps = 1000,
        save_steps = 1000,
        output_dir = settings_comet["train_parameters"]["output_dir"],
        logging_dir = settings_comet["train_parameters"]["logging_dir"],
        per_device_train_batch_size = settings_comet["train_parameters"]["per_device_train_batch_size"],
        per_device_eval_batch_size = settings_comet["train_parameters"]["per_device_eval_batch_size"],
        learning_rate = settings_comet["train_parameters"]["learning_rate"],
        weight_decay = settings_comet["train_parameters"]["weight_decay"],
        num_train_epochs = settings_comet["train_parameters"]["num_train_epochs"],
        save_total_limit = 5,
        load_best_model_at_end = True,
        report_to = "wandb",
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForComet(
        tokenizer = tokenizer,
        mlm = False
    )

    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = datasets_dict["train"],
        eval_dataset = datasets_dict["validation"],
        data_collator = data_collator,
        tokenizer = tokenizer
    )

    trainer.train()



if __name__ == "__main__":
    """
    nohup python scripts/comet/train.py --device_id "2, 3" --patent_domain 情報系 --run_name trial &
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", type=str)
    parser.add_argument("--patent_domain", type=str, default="情報系")
    parser.add_argument("--run_name", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    os.environ["WANDB_PROJECT"]= f"COMET_patent_{args.patent_domain}"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    with open("./settings_comet.json", "r") as f:
        settings_comet = json.load(f)
    
    with open("./settings_data.json", "r") as f:
        settings_data = json.load(f)

    wandb.init(project=f"COMET_patent_{args.patent_domain}", run_name=args.run_name)
    main(settings_comet, settings_data, args.patent_domain)
    wandb.finish()