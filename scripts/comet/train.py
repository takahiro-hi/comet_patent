import json, sys, datasets, argparse, os, wandb

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    Trainer
)
from transformers.integrations import WandbCallback

sys.path.append("./scripts/utils")
from load_data import load_gold, load_silver


ADV_TOKEN = "xEffect"


def to_input_format(head, tail, tokenizer):

    return head + ADV_TOKEN + tail + tokenizer.eos_token



def load_dataset(settings, args_cli, tokenizer):

    gold_data = load_gold(settings["data"], "patent", args_cli.patent_domain)
    silver_data = load_silver(settings["data"], settings["comet"]["train_parameters"]["tr_silver_temperature"], "patent", args_cli.patent_domain)

    datasets_dict = datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"data": [to_input_format(d[0], d[1], tokenizer) for d in silver_data[:int(len(silver_data) * 0.9)]]}),
        "validation": datasets.Dataset.from_dict({"data": [to_input_format(d[0], d[1], tokenizer) for d in silver_data[int(len(silver_data) * 0.9):]]}),
        "test": datasets.Dataset.from_dict({"data": [to_input_format(d[0], d[1], tokenizer) for d in gold_data]})
    })

    print(f"train: {len(datasets_dict['train'])}, validation: {len(datasets_dict['validation'])}, test: {len(datasets_dict['test'])}")

    tokenized_datasets = datasets_dict.map(
        lambda examples: tokenizer(
            examples["data"],
            truncation = True,
            max_length = 256,
        ),
        batched = True,
        remove_columns = datasets_dict["train"].column_names
    )

    test_data = {
        "inputs": [d[0] + ADV_TOKEN for d in gold_data],
        "outputs": [d[1] + tokenizer.eos_token for d in gold_data]
    }

    return tokenized_datasets, test_data



class DataCollatorForComet(DataCollatorForLanguageModeling):

    def torch_call(self, examples):

        batch = super().torch_call(examples)            
        labels = batch["labels"]

        rel_mask = labels >= self.tokenizer.vocab_size
        tail_mask = (rel_mask.cumsum(dim=-1) - rel_mask.to(int)).to(bool)
        labels[~tail_mask] = -100

        batch["labels"] = labels

        return batch



class CustomWandbCallback(WandbCallback):

    def __init__(self, tokenizer, eval_data, num_samples=50):

        super().__init__()
        self.tokenizer = tokenizer
        self.test_dataset = {
            "inputs": eval_data["inputs"][:num_samples],
            "outputs": eval_data["outputs"][:num_samples]
        }
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.adv_id = self.tokenizer.convert_tokens_to_ids(ADV_TOKEN)
    

    def on_evaluate(self, args, state, control, model, **kwargs):

        super().on_evaluate(args, state, control, **kwargs)

        table = wandb.Table(columns=["index", "head+rel", "tail_pred", "tail_gold"])

        input_ids = self.tokenizer(self.test_dataset["inputs"], truncation=True, max_length=256, return_tensors="pt", padding=True)
        output_ids = model.generate(**input_ids.to("cuda"))
        filtered_output_ids = [
            [token_id for token_id in seq if not (token_id in self.special_ids and token_id != self.adv_id)]
            for seq in output_ids.tolist()
        ]
        outputs = [self.tokenizer.decode(seq, skip_special_tokens=False) for seq in filtered_output_ids]

        for idx, (gold_i, gold_o, pred) in enumerate(zip(self.test_dataset["inputs"], self.test_dataset["outputs"], outputs)):
            head, tail = pred.split(ADV_TOKEN)
            table.add_data(idx, gold_i, tail, gold_o)
        
        wandb.log({"gold_test_table": table, "step": state.global_step})



def main(settings, args_cli):

    settings_comet_tr = settings["comet"]["train_parameters"]

    model = AutoModelForCausalLM.from_pretrained(settings_comet_tr["model_name"]).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(settings_comet_tr["model_name"], padding_side="left")
    datasets_dict, test_data = load_dataset(settings, args_cli, tokenizer)

    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer), "not added properly (1)"
    assert len(tokenizer.encode(ADV_TOKEN, add_special_tokens=False)) == 1, "not added properly (2)"

    interval = 1000
    args_trainer = TrainingArguments(
        do_train = True,
        do_eval = True,
        eval_strategy = "steps",
        eval_steps = interval,
        logging_steps = interval,
        save_steps = interval,
        output_dir = settings_comet_tr["output_dir"],
        logging_dir = settings_comet_tr["logging_dir"],
        per_device_train_batch_size = settings_comet_tr["per_device_train_batch_size"],
        per_device_eval_batch_size = settings_comet_tr["per_device_eval_batch_size"],
        learning_rate = settings_comet_tr["learning_rate"],
        weight_decay = settings_comet_tr["weight_decay"],
        num_train_epochs = settings_comet_tr["num_train_epochs"],
        save_total_limit = 5,
        load_best_model_at_end = True,
        report_to = ["wandb", "tensorboard"],
        run_name = args_cli.run_name
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForComet(
        tokenizer = tokenizer,
        mlm = False
    )

    trainer = Trainer(
        model = model,
        args = args_trainer,
        train_dataset = datasets_dict["train"],
        eval_dataset = datasets_dict["validation"],
        data_collator = data_collator
    )

    trainer.add_callback(CustomWandbCallback(
        tokenizer = tokenizer,
        eval_data = test_data
    ))

    trainer.train()



if __name__ == "__main__":
    """
    nohup python scripts/comet/train.py --device_id "2, 3" --patent_domain 情報系 --run_name trial &
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", type=str)
    parser.add_argument("--patent_domain", type=str, default="情報系")
    parser.add_argument("--run_name", type=str)
    args_cli = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args_cli.device_ids
    os.environ["WANDB_PROJECT"]= f"COMET_patent_{args_cli.patent_domain}"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    with open("./settings_comet.json", "r") as f:
        settings_comet = json.load(f)
    with open("./settings_data.json", "r") as f:
        settings_data = json.load(f)
    settings = {"comet": settings_comet, "data": settings_data}

    wandb.init(
        project = f"COMET_patent_{args_cli.patent_domain}", 
        name = args_cli.run_name,
        config = settings["comet"],
        dir = settings["comet"]["train_parameters"]["wandb_dir"]
    )

    main(settings, args_cli)

    wandb.finish()