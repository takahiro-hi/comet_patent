from transformers import AutoModelForCausalLM, AutoTokenizer



ADV_TOKEN = "xEffect"
device = "cpu"

model = "./result_化学系/comet/adv_init_1_no2/model"

tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/comet-v2-gpt2-small-japanese", padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model).to(device)

special_ids = set(tokenizer.all_special_ids)
adv_id = tokenizer.convert_tokens_to_ids(ADV_TOKEN)

input_data = ["触媒反応を適用" + ADV_TOKEN]
input_ids = tokenizer(input_data, truncation=True, max_length=256, return_tensors="pt", padding=True)
output_ids = model.generate(**input_ids.to(device))

print(f"output text (not skip special tokens): {tokenizer.decode(output_ids[0], skip_special_tokens=False)}")
