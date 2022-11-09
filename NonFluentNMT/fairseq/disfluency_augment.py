import argparse
import tqdm
import numpy as np
import random
import math
import difflib
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
'''
汉语不流利现象参考自 https://github.com/dqqcasia/Translation_Disfluency_Detection
file_path文件为bpe之后的文件
filler_file文件为bpe之后的文件，其中codec需要和file_path的相同
'''
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", "-f", default="")
parser.add_argument("--save_path", "-s", default="")
parser.add_argument("--label_save_path", "-ls", default="")
parser.add_argument("--type", "-t", default="") # 1 for interregnum, 2 for restart and repair, 3 for repetition
parser.add_argument("--filler_file", default="")
parser.add_argument("--propotion", type=float, default=0)
parser.add_argument("--pretrain_model_name", default="")
parser.add_argument("--Batch_size", type=int, default=400)
parser.add_argument("--device", type=int, default=None)
args = parser.parse_args()

registered_rules = {}

def register_rule(rule_func):
    registered_rules[rule_func.__name__] = rule_func
    return rule_func


@register_rule
def rule_1(args, main_files, label_files):
    # generate filler words
    result_line = []
    result_label = []
    with open(args.filler_file, "r", encoding="utf-8") as f:
        fillers = f.readlines()
    for main_line, llabel in tqdm.tqdm(zip(main_files, label_files)):
        tokens = main_line.strip().split()
        labels = llabel.strip().split()
        try:
            if args.propotion > 0:
                f_prop = round(len(tokens) * args.propotion)
                ids = random.sample(range(0, len(tokens)), f_prop)
            else:
                ids = random.sample(range(0, len(tokens)), np.random.randint(1, 3))
        except:
            ids = random.sample(range(0, len(tokens)), np.random.randint(1, len(tokens)+1))
        items = [i.strip() for i in np.random.choice(fillers, size=len(ids))]
        ii = 0
        labels_ = []
        line_ = []
        for i in range(len(tokens)):
            if i in ids:
                line_ += items[ii].split(" ") + [tokens[i]]
                labels_ += ["1"] * len(items[ii].split(" ")) + [labels[i]]
                ii += 1
            else:
                line_ += [tokens[i]]
                labels_ += [labels[i]]
        assert len(line_) == len(labels_)
        result_line.append(" ".join(line_))
        result_label.append(" ".join(labels_))
    return result_line, result_label


@register_rule
def rule_2(args, main_files, label_files):
    # generate restart and repair
    result_line = []
    result_label = []
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.pretrain_model_name)
    FM_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=5, device=args.device)
    raw_text_list_1 = [i.strip().lower() for i in main_files]
    raw_label_list_1 = [i.strip() for i in label_files]
    masked_token_str_list = []
    batches_sentences = []
    batches_labels = []
    text = []
    label = []
    count = 0
    Batch_size = args.Batch_size
    # batch
    for text_, label_ in zip(raw_text_list_1, raw_label_list_1):
        text.append(text_)
        label.append(label_)
        count += 1
        if count % Batch_size == 0:
            batches_sentences.append(text)
            batches_labels.append(label)
            text = []
            label = []
        elif count % Batch_size != 0 and count == len(raw_text_list_1):
            batches_sentences.append(text)
            batches_labels.append(label)

    texts = []
    text_ = []
    text = []
    mask_positions = []
    for batch_sentences in batches_sentences:
        for i in batch_sentences:
            tokens = i.strip().split()
            p = np.random.randint(0, len(tokens))
            mask_positions.append(p)
            for j in range(len(tokens)):
                if j != p:
                    text.append(tokens[j])
                else:
                    p_1 = np.random.randint(0, len(tokens[j]))
                    t_1 = ""
                    for k in range(len(tokens[j])):
                        if k != p_1:
                            t_1 += tokens[j][k]
                        else:
                            t_1 += "[MASK]"
                    text.append(t_1)
            text_.append(" ".join(text))
            text = []
        texts.append(text_)
        text_ = []

    results_list = []
    for i in tqdm.tqdm(texts): # 每个batch
        results = FM_pipeline(i)
        results_list.append(results)

    with open(args.filler_file, encoding="utf-8") as f:
        filler_candidate_list = [i.strip() for i in f.readlines()]

    for b in range(len(results_list)):
        for i in range(len(results_list[b])):
            sample_id = np.random.randint(0, 5)
            mask_token = results_list[b][i][sample_id]["token_str"]  # 海
            mask_word = re.findall("(\S*\[MASK\]\S*)", texts[b][i])[0].replace("[MASK]", mask_token)
            repair_sentence_list = batches_sentences[b][i].split()
            filler_candidate = random.sample(filler_candidate_list, 1)[0]
            filler_id = np.random.choice(3, p=[0.4, 0.4, 0.2])
            filler = [" ，", " ， " + filler_candidate, ""][filler_id]
            repair_sentence_list.insert(mask_positions[b*args.Batch_size+i], mask_word + filler)
            repair_sentence = " ".join(repair_sentence_list)

            repair_label_list = batches_labels[b][i].strip().split()
            filler_len = [1, 1 + len(filler_candidate.split()), 0]
            repair_label_list.insert(mask_positions[b*args.Batch_size+i], "1")
            if filler_id != 2:
                for j in range(filler_len[filler_id]):
                    repair_label_list.insert(mask_positions[b*args.Batch_size+i], "1")
            repair_label = " ".join(repair_label_list)
            assert len(repair_label.strip().split()) == len(repair_sentence.strip().split())
            result_line.append(repair_sentence)
            result_label.append(repair_label)
    return result_line, result_label


@register_rule
def rule_3(args, main_files, label_files):
    # generate repetiton
    result_line = []
    result_label = []
    with open(args.filler_file, "r", encoding="utf-8") as f:
        fillers = f.readlines()
    for main_line, llabel in tqdm.tqdm(zip(main_files, label_files)):
        tokens = main_line.strip().split()
        lllabels = llabel.strip().split()
        try:
            if args.propotion > 0:
                f_prop = round(len(tokens) * args.propotion)
                ids = random.sample(range(0, len(tokens)-1), f_prop)
            else:
                ids = random.sample(range(0, len(tokens)-1), np.random.randint(1, 4))
        except:
            ids = []
        ids_1 = []
        ids_2 = []
        for i in ids:
            if random.random() < 0.7:
                ids_1.append(i)
            else:
                ids_2.append(i)
        label = []
        line_ = []
        for i in range(len(tokens)):
            if i in ids_1:
                r_2 = np.random.randint(1, 3)  # 重复长度
                if random.random() < 0.5:
                    line_ += tokens[i: i + r_2] + [" ,"] + [tokens[i]]
                    label += ["1"] * (r_2 + 1)  + [lllabels[i]]
                else:
                    filler = fillers[np.random.randint(0,len(fillers))].strip().split(" ")
                    line_ += tokens[i: i + r_2] + [" ,"] + filler + [tokens[i]]
                    label += ["1"] * (r_2 + 1 + len(filler)) + [lllabels[i]]
            elif i in ids_2:
                if random.random() < 0.5:
                    try:
                        line_ += [tokens[i][:np.random.randint(1, len(tokens[i]))] + "-"] + [" ,"] + [tokens[i]]
                    except:
                        line_ += [tokens[i] + "-"] + [" ,"] + [tokens[i]]
                    label += ["1"] * 2 + [lllabels[i]]
                else:
                    line_ += [tokens[i]] * 2
                    label += ["1"] + [lllabels[i]]
            else:
                line_ += [tokens[i]]
                label += [lllabels[i]]
        assert len(line_) == len(label)
        result_line.append(" ".join(line_))
        result_label.append(" ".join(label))
    return result_line, result_label


def main():
    with open(args.file_path, mode="r", encoding="utf-8") as f:
        main_files = f.readlines()
    label_files = [" ".join(["0"] * len(i.strip().split())) for i in main_files]
    for i in args.type.strip().split(","):
        main_files, label_files = registered_rules["rule_" + i](args, main_files, label_files)
        all_prop = []
        for i in label_files:
            all_prop.extend(i.strip().split())
        print(all_prop.count("1") / len(all_prop))
        assert len(main_files) == len(label_files)
    with open(args.label_save_path, mode="w", encoding="utf-8") as l, open(args.save_path, mode="w", encoding="utf-8") as s:
        for line, label in zip(main_files, label_files):
            s.write(line + "\n")
            l.write(label + "\n")


if __name__ == '__main__':
    main()
