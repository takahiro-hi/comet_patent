import torch, uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


"""
uvicorn main:app --reload
"""

class RequestData(BaseModel):
    text: str


app = FastAPI()

device = "cuda:3"
model_path = "../../result_化学系/comet/adv_init_1_no2/model"

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/comet-v2-gpt2-small-japanese", padding_side="left")

ADV_TOKEN = "xEffect"
assert len(tokenizer.encode(ADV_TOKEN, add_special_tokens=False)) == 1, "ADV_TOKEN should be one token"


@app.post("/generate")
def generate_text(data: RequestData):

    input_text = [data.text + ADV_TOKEN]
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**input_ids.to(device))

    print(output_ids)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    generated_text = output[len(data.text):]
    assert data.text == output[:len(data.text)], "Input text does not match the beginning of the generated text"

    return {"generated_text": generated_text}


if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)