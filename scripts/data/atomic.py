import os, json
import pandas as pd




def get_xeffect(data_df):

    ret_data = []

    for idx, row in data_df.iterrows():
        for d in row["inference"]["event"]["after"]:
            ret_data.append((row["event"], d))
    
    return ret_data



def main(settings_data):

    df = pd.read_json(os.path.join(settings_data["dir_path"]["atomic"]["gold"], "graph_v2_mrph.jsonl"), orient="records", lines=True)
    effect_data = get_xeffect(df)
    print(len(effect_data))

    with open(os.path.join(settings_data["dir_path"]["atomic"]["gold"], "xeffect.json"), "w") as f:
        f.write(json.dumps(effect_data, indent=4, ensure_ascii=False))





if __name__ == "__main__":

    with open("./settings_data.json", "r") as f:
        settings_data = json.load(f)

    main(settings_data)