import argparse
import json
import re

import cn2an
from sklearn.metrics import precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset_path", default="data/testset/cjo22/testset.json")
    parser.add_argument(
        "--resp_file",
        default="data/output/llm_out/cjo22/CNN/article/3shot/qwen3-max.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only evaluate first N samples; 0 means all",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data = []
    with open(args.testset_path, encoding="utf8") as f:
        for line in f:
            data.append(json.loads(line))

    results = []
    with open(args.resp_file, encoding="utf8") as f:
        for line in f:
            results.append(json.loads(line))

    n = len(data)
    if args.limit and args.limit > 0:
        n = min(args.limit, len(data), len(results))
    else:
        n = min(len(data), len(results))

    data = data[:n]
    results = results[:n]

    y_true = [int(max(case["meta"]["relevant_articles"])) for case in data]

    y_pred = []
    for obj in results:
        resp = obj["choices"][0]["message"]["content"]
        res = re.findall(r"第(.*?)条", resp)
        ars = [0]
        for i in res:
            if i.isdigit():
                ars.append(int(i))
            else:
                try:
                    ars.append(int(cn2an.cn2an(i, mode="smart")))
                except Exception:
                    ars.append(0)
        y_pred.append(max(ars))

    acc, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
    map_, mar, maf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    print(
        f"Samples:{n}, acc:{acc * 100:.2f}, map:{map_ * 100:.2f}, mar:{mar * 100:.2f}, maf:{maf * 100:.2f}"
    )


if __name__ == "__main__":
    main()
