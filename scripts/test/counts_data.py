import os, json, random, sys

sys.path.append("./scripts/train")
from utils_load_data import to_input_format, load_gold, load_silver




if __name__=="__main__":

    random.seed(42)

    with open("./settings.json", "r") as f:
        config = json.load(f)

    gold_atomic = load_gold(config, "atomic")
    gold_patent = load_gold(config, "patent", "情報系")

    silver_atomic = load_silver(config, "atomic")
    silver_patent = load_silver(config, "patent", "情報系")

    print("size of gold_atomic:", len(gold_atomic))
    print("size of gold_patent:", len(gold_patent))
    print("size of silver_atomic:", len(silver_atomic))
    print("size of silver_patent:", len(silver_patent))
          