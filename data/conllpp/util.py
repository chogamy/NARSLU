import os
from ordered_set import OrderedSet

from numpy import average

DIR = os.path.dirname(os.path.realpath(__file__))


def load_data(path):
    tokens = []
    labels = []
    token = []
    label = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "-DOCSTART- -X- -X- O":
                continue

            if line == "":
                if len(token) > 0:
                    tokens.append(token)
                    labels.append(label)
                token = []
                label = []

            else:
                t, _, _, l = line.split(" ")
                token.append(t)
                label.append(l)

    return tokens, labels


def make(tokens, labels, file_path):
    with open(os.path.join(file_path, "seq.in"), "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(" ".join(token) + "\n")

    with open(os.path.join(file_path, "seq.out"), "w", encoding="utf-8") as f:
        for label in labels:
            f.write(" ".join(label) + "\n")


if __name__ == "__main__":
    data = "train"
    path = os.path.join(DIR, data, f"{data}.txt")
    tokens, labels = load_data(path)
    make(tokens, labels, os.path.join(DIR, data))

    if data == "train":
        flatten_list = []
        for label in labels:
            flatten_list += label

        labels = ["PAD", "UNK", "MASK"] + list(OrderedSet(flatten_list))

        with open(os.path.join(DIR, "slot_label.txt"), "w", encoding="utf-8") as f:
            for label in labels:
                f.write(label + "\n")
