


def tokenize_data(data, tokenizer, device):

    return tokenizer(data, return_tensors='pt', padding="longest").to(device)