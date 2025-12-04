import argparse
import json
from collections import Counter

from sklearn.metrics import precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset_path", default="data/testset/cjo22/testset.json")
    parser.add_argument(
        "--resp_file",
        default="data/output/llm_out/cjo22/CNN/charge/3shot/qwen3-max.json",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="only evaluate first N samples; 0 means all"
    )
    return parser.parse_args()


laic_charge_match = {
    "非法生产、买卖、运输制毒物品、走私制毒物品": [
        ("生产", "制毒物品"),
        ("买卖", "制毒物品"),
        ("运输", "制毒物品"),
        ("走私", "制毒物品"),
    ],
    "非法经营": [("非法", "经营")],
    "非法转让、倒卖土地使用权": [("转让", "土地"), ("倒卖", "土地")],
}

cail_charge_match = {
    "重大劳动安全事故": [("劳动", "事故")],
    "容留他人吸毒": [("容留", "吸毒")],
    "非法种植毒品原植物": [("毒品",)],
    "盗伐林木": [("盗伐", "林木")],
    "故意杀人": [("故意", "杀人")],
    "交通肇事": [("肇事",)],
    "污染环境": [("污染",)],
    "强奸": [("强奸",)],
    "合同诈骗": [("合同", "诈骗")],
    "生产、销售不符合安全标准的食品": [("食品", "安全")],
    "强制猥亵、侮辱妇女": [("猥亵",), ("侮辱", "妇女")],
    "妨害信用卡管理": [("信用卡", "管理")],
    "赌博": [("赌博",)],
    "生产、销售伪劣产品": [("伪劣", "产品")],
    "妨害公务": [("妨害", "公务")],
    "职务侵占": [("职务", "侵占")],
    "非法采矿": [("采矿",)],
    "滥用职权": [("滥用", "职权")],
    "破坏广播电视设施、公用电信设施": [("破坏", "广播"), ("破坏", "电信")],
    "放火": [("放火",)],
    "伪造、变造、买卖国家机关公文、证件、印章": [("伪造", "印章"), ("伪造", "公文")],
    "非法采伐、毁坏国家重点保护植物": [("保护植物",)],
    "开设赌场": [("开设", "赌场")],
    "生产、销售假药": [("假药",)],
    "非法吸收公众存款": [("公众", "存款")],
    "玩忽职守": [("玩忽", "职守")],
}

total_charge_match = {**laic_charge_match, **cail_charge_match}


def get_similar_charge(text, charge_similar):
    contain_charge_set = set()
    for c, precedent_lst in charge_similar.items():
        for precedent_words in precedent_lst:
            if sum(w in text for w in precedent_words) == len(precedent_words):
                contain_charge_set.add(c)
    if not contain_charge_set:
        return "#"
    charge_set = sorted(contain_charge_set, key=lambda x: len(x), reverse=True)
    return charge_set[0]


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

    charges = [case["meta"]["accusation"][0] for case in data]
    c_set = set(charges)

    y_true = charges
    y_pred = []
    for output in results:
        text = output["choices"][0]["message"]["content"]
        cur_c = "#"
        contain_charge_set = {c for c in c_set if c in text}
        if len(contain_charge_set) == 1:
            cur_c = list(contain_charge_set)[0]
        if cur_c == "#":
            cur_c = get_similar_charge(text, total_charge_match)
        y_pred.append(cur_c)

    acc, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    print(
        f"Samples:{n}, acc:{acc * 100:.2f}, mp:{mp * 100:.2f}, mr:{mr * 100:.2f}, mf:{mf * 100:.2f}"
    )


if __name__ == "__main__":
    main()
