import re

from collections import defaultdict





def load_row_ann(ann_file_path):

    tag_data, rel_data = defaultdict(dict), defaultdict(dict)

    # load tag
    for line in open(ann_file_path, "r"):
        line = line.strip()
        if line.startswith("T"):
            tag_id, tag_info, content = line.split("\t")
            if ";" in tag_info:
                assert tag_info.count(";") == 1, f"technical debt: tag_info has unexpected semicolon ({ann_file_path})"
                tag_type, s1, e1, s2, e2 = re.split('[ ;]', tag_info)
                assert content.count(" ") == 1, f"technical debt: content has unexpected space ({ann_file_path})"
                content = content.replace(" ", "")
            else:
                tag_type, s1, e1 = tag_info.split(" ")
                s2, e2 = -1, -1

            tag_data[tag_id] = {
                "type": tag_type,
                "loc1": (int(s1), int(e1)),
                "loc2": (int(s2), int(e2)),
                "content": content
            }
    
    # load relation
    for line in open(ann_file_path, "r"):
        line = line.strip()
        if line.startswith("R"):
            rel_id, rel_info = line.split("\t")
            rel_type, arg1, arg2 = rel_info.split(" ")
            rel_data[rel_id] = {
                "type": rel_type,
                "arg1": arg1.replace("Arg1:", "", 1),
                "arg2": arg2.replace("Arg2:", "", 1)
            }
        
    return tag_data, rel_data