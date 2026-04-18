import os
import sys
import json
import re
from tqdm import tqdm

# from NSM.data.basic_dataset import BasicDataLoader

def tokenize_sent(question_text):
        question_text = question_text.strip().lower()
        question_text = re.sub('\'s', ' s', question_text)
        words = []
        for w_idx, w in enumerate(question_text.split(' ')):
            w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
            if w == '':
                continue
            words += [w]
        return words

def load_vocab(filename):
    f = open(filename)
    voc2id = {}
    for line in f:
        line = line.strip()
        voc2id[line] = len(voc2id)
    return voc2id


def load_vocab_json(filename):
    """Load vocabulary from JSON file containing a list of relations/entities"""
    with open(filename, encoding='utf-8') as f:
        relations = json.load(f)
    voc2id = {}
    for rel in relations:
        voc2id[rel] = len(voc2id)
    return voc2id


def output_dict(ele2id, filename):
    num_ele = len(ele2id)
    id2ele = {v: k for k, v in ele2id.items()}
    f = open(filename, "w")
    for i in range(num_ele):
        f.write(id2ele[i] + "\n")
    f.close()


def deal_rel(tp_str, dataset):
    if dataset.lower() == "cwq" or dataset.lower() == "webqsp":
        return tp_str.split(".")
    elif dataset.lower() == "metaqa":
        return [tp_str]
    else:
        raise NotImplementedError


def add_word_in_Relation(relation_file, vocab, dataset="webqsp"):
    # Auto-detect file format based on extension
    if relation_file.endswith('.json'):
        rel2id = load_vocab_json(relation_file)
    else:
        rel2id = load_vocab(relation_file)
    
    max_len = 0
    oov = set()
    for rel in rel2id:
        domain_list = deal_rel(rel, dataset)
        tp_list = []
        for domain_str in domain_list:
            tp_list += domain_str.split("_")
        # print(tp_list)
        words = []
        for w_idx, w in enumerate(tp_list):
            w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
            if w != '' and w not in vocab:
                vocab[w] = len(vocab)
                oov.add(w)
                words.append(vocab[w])
        if len(words) > max_len:
            max_len = len(words)
    print("Max length:", max_len)
    print("Total", len(vocab))
    print("OOV relation", len(oov))
    return vocab


def add_word_in_question(inpath):
    vocab = {}
    for split in ["train_subgraph", "dev_subgraph", "test_subgraph"]:
        infile = os.path.join(inpath, split + ".json")
        if not os.path.exists(infile):
            print(f"Warning: File {infile} not found, skipping.")
            continue
        with open(infile) as f:
            data = json.load(f)
        for obj in data:
            tp_obj = obj
            questions = tp_obj["question_nl"]
            for question in questions:
                tokens = tokenize_sent(question)
                for j, word in enumerate(tokens):
                    if word not in vocab:
                        vocab[word] = len(vocab)

        # print(split, len(vocab))
    # out_file = os.path.join(outpath, "vocab.txt")
    # output_dict(vocab, out_file)
    return vocab



inpath = sys.argv[1]
outpath = sys.argv[2]
dataset = sys.argv[3]
question_vocab = add_word_in_question(inpath)
relation_file = os.path.join(inpath, "relations.txt")
full_vocab = add_word_in_Relation(relation_file, question_vocab, dataset)
out_file = os.path.join(outpath, "vocab_new.txt")
output_dict(full_vocab, out_file)