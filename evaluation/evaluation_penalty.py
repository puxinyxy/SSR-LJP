import argparse
import json

from sklearn.metrics import precision_recall_fscore_support

pt_cls2str = [
    "其他",
    "六个月以内",
    "六到九个月",
    "九个月到一年",
    "一到两年",
    "二到三年",
    "三到五年",
    "五到七年",
    "七到十年",
    "十年以上",
]


def get_pt_cls(pt):
    if pt > 10 * 12:
        pt_cls = 9
    elif pt > 7 * 12:
        pt_cls = 8
    elif pt > 5 * 12:
        pt_cls = 7
    elif pt > 3 * 12:
        pt_cls = 6
    elif pt > 2 * 12:
        pt_cls = 5
    elif pt > 1 * 12:
        pt_cls = 4
    elif pt > 9:
        pt_cls = 3
    elif pt > 6:
        pt_cls = 2
    elif pt > 0:
        pt_cls = 1
    else:
        pt_cls = 0
    return pt_cls


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset_path", default="data/testset/cjo22/testset.json")
    parser.add_argument(
        "--resp_file",
        default="data/output/llm_out/cjo22/CNN/penalty/3shot/qwen3-max.json",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="only evaluate first N samples; 0 means all"
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

    n = min(len(data), len(results))
    if args.limit and args.limit > 0:
        n = min(n, args.limit)
    data = data[:n]
    results = results[:n]

    y_true_raw = [case["meta"]["term_of_imprisonment"]["imprisonment"] for case in data]
    y_true = [pt_cls2str[get_pt_cls(p)] for p in y_true_raw]

    y_pred = []
    for obj in results:
        text = obj["choices"][0]["message"]["content"]
        pred_p = ""
        for p_str in pt_cls2str:
            if p_str in text:
                pred_p = p_str
        y_pred.append(pred_p)

    acc, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
    map_, mar, maf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    print(
        f"Samples:{n}, acc:{acc * 100:.2f}, map:{map_ * 100:.2f}, mar:{mar * 100:.2f}, maf:{maf * 100:.2f}"
    )


if __name__ == "__main__":
    main()
