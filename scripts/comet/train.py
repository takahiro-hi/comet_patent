import json, sys, datasets, argparse, os, wandb, sys, random

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    Trainer,
    EarlyStoppingCallback
)
from transformers.integrations import WandbCallback

sys.path.append("./scripts/utils")
from load_data import load_gold, get_settings


ADV_TOKEN = "xEffect"



def load_silver(args_cli, settings, run_name):

    if run_name == "all":
        path = os.path.join(settings["data"]["dir_path"]["patent"]["train"], f'{args_cli.patent_domain}_triple_{settings["tr_params"]["tr_silver_temperature"]}.json')
    elif run_name == "base":
        path = os.path.join(settings["data"]["dir_path"]["patent"]["train"], f'{args_cli.patent_domain}_triple_{settings["tr_params"]["tr_silver_temperature"]}_base.json')
    else:
        path = os.path.join(settings["data"]["dir_path"]["patent"]["train"], f'{args_cli.patent_domain}_triple_{settings["tr_params"]["tr_silver_temperature"]}_adv_{args_cli.adv_model}.json')

    with open(path, "r") as f:
        data = json.load(f)

    random.shuffle(data)
    return data


def to_input_format(head, tail, tokenizer):

    return head + ADV_TOKEN + tail + tokenizer.eos_token



def load_dataset(settings, args_cli, tokenizer, run_name):

    gold_data = load_gold(settings["data"], "patent", args_cli.patent_domain, True)
    silver_data = load_silver(args_cli, settings, run_name)

    datasets_dict = datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"data": [to_input_format(d[0], d[1], tokenizer) for d in silver_data[:int(len(silver_data) * 0.9)]]}),
        "validation": datasets.Dataset.from_dict({"data": [to_input_format(d[0], d[1], tokenizer) for d in silver_data[int(len(silver_data) * 0.9):]]}),
    })

    print(f"train: {len(datasets_dict['train'])}, validation: {len(datasets_dict['validation'])}")

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



def main(settings, args_cli, result_path, run_name):

    params_comet_tr = settings["tr_params"]

    model = AutoModelForCausalLM.from_pretrained(params_comet_tr["model_name"]).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(params_comet_tr["model_name"], padding_side="left")
    datasets_dict, test_data = load_dataset(settings, args_cli, tokenizer, run_name)

    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer), "not added properly (1)"
    assert len(tokenizer.encode(ADV_TOKEN, add_special_tokens=False)) == 1, "not added properly (2)"

    interval = 1000
    args_trainer = TrainingArguments(
        do_train = True,
        do_eval = True,
        eval_strategy = "steps",
        eval_steps = interval,
        logging_steps = interval,
        save_strategy = "steps",
        save_steps = interval,
        output_dir = result_path,
        logging_dir = os.path.join(result_path, "logs"),
        per_device_train_batch_size = params_comet_tr["per_device_train_batch_size"],
        per_device_eval_batch_size = params_comet_tr["per_device_eval_batch_size"],
        learning_rate = params_comet_tr["learning_rate"],
        weight_decay = params_comet_tr["weight_decay"],
        num_train_epochs = params_comet_tr["num_train_epochs"],
        save_total_limit = 5,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        report_to = ["wandb", "tensorboard"],
        run_name = run_name
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

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience = 5,
        early_stopping_threshold = 0.01
    )
    trainer.add_callback(early_stopping_callback)

    trainer.train()

    trainer.save_model(os.path.join(result_path, "model"))
    trainer.save_state()



if __name__ == "__main__":

    """
    nohup python scripts/comet/train.py --device_id "1" --patent_domain "化学系" --run_name "all" > nohup_all.out &
    nohup python scripts/comet/train.py --device_id "2" --patent_domain "化学系" --run_name "base" > nohup_base.out &
    nohup python scripts/comet/train.py --device_id "3" --patent_domain "化学系" --run_name "adv" --adv_model "init_1_no1" > nohup_adv.out &
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", type=str)
    parser.add_argument("--patent_domain", type=str, default="情報系")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--adv_model", type=str, default=None)
    args_cli = parser.parse_args()

    if args_cli.run_name in ["all", "base"]:
        run_name = args_cli.run_name
    elif args_cli.run_name == "adv":
        run_name = f"{args_cli.run_name}_{args_cli.adv_model}"

    os.environ["CUDA_VISIBLE_DEVICES"] = args_cli.device_ids
    os.environ["WANDB_PROJECT"]= f"COMET_patent_{args_cli.patent_domain}"
    os.environ["WANDB_LOG_MODEL"] = "end"

    settings = get_settings("comet")
    result_path = os.path.join(settings["result_path"][args_cli.patent_domain], f"comet/{run_name}")

    wandb.init(
        project = f"COMET_patent_{args_cli.patent_domain}", 
        name = run_name,
        config = settings["tr_params"],
        dir = os.path.join(result_path, "wandb")
    )

    main(settings, args_cli ,result_path, run_name)

    wandb.finish()