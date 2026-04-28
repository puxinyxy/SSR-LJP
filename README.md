# Agentic_Rag_LJP
my ljp code

## CAIL2018 璇勬祴鍛戒护

```bash
python ljp_eval_cail2018.py --limit 100 --offset 0 --top-k 5 --dataset-path data/testset/test_sampled_single_seed_42.json --candidates-path data/candidates/precedents_cail.json --output-dir output_cail2018
```

鍙傛暟璇存槑锛?
- `--limit 100`锛氳瘎娴嬪墠 100 鏉℃牱鏈€?- `--offset 0`锛氫粠绗?0 鏉℃牱鏈紑濮嬨€?- `--top-k 5`锛氭绱㈤樁娈佃繑鍥?top-5 鍊欓€夈€?- `--dataset-path ...`锛氭寚瀹?CAIL2018 娴嬭瘯闆嗘枃浠惰矾寰勩€?- `--candidates-path ...`锛氭寚瀹?CAIL2018 鍏堜緥鍊欓€夋枃浠惰矾寰勩€?- `--output-dir output_cail2018`锛氳瘎娴嬬粨鏋滆緭鍑虹洰褰曪紙浼氱敓鎴愭椂闂存埑鍛藉悕鐨?`.jsonl` 鍜?`.xlsx` 鏂囦欢锛夈€?
## calc_ours 鎸囦护

```bash
python calc_ours.py --input output_cail2018\ljp_eval_cail2018_20260309_210549.jsonl
# cjo22 + LOT_prompt 缁撴灉鎸囨爣缁熻
python calc_ours.py --input G:\graduate_1\Code\Camel\baseline_output\plain_prompt_cjo22_qwen3-max_LOT_prompt_20260330_194138\LOT_prompt_results_qwen3-max.jsonl
```

鍙傛暟璇存槑锛?
- `calc_ours.py`锛氬妯″瀷棰勬祴缁撴灉杩涜鎸囨爣璁＄畻锛堝 law/acc/penalty 鐩稿叧鎸囨爣锛夈€?- `--input ...jsonl`锛氭寚瀹氳瘎娴嬭緭鍑虹殑 JSONL 鏂囦欢璺緞锛岃剼鏈細璇诲彇璇ユ枃浠跺苟鎵撳嵃缁熻缁撴灉銆?
## eval_llm_rag_law 鎸囦护

```bash
python eval_llm_rag_law.py --dataset_path data/testset/test_cail_sampled_single_seed_42.json --law_dir data/meta/laws.txt --limit 100 --offset 0 --topk_law 3 --output_path embedding_output/llm_rag_law_cail.jsonl
# cjo22 full testset
python eval_llm_rag_law.py --dataset_path data/testset/testset.json --law_dir data/meta/laws.txt --limit 1698 --offset 0 --topk_law 3
```

鍙傛暟璇存槑锛?
- `eval_llm_rag_law.py`锛氭墽琛屾硶鏉￠娴嬬殑 LLM+RAG 鍩虹嚎璇勬祴銆?- `--dataset_path ...`锛氭寚瀹氬緟璇勬祴鏁版嵁闆嗚矾寰勶紙杩欓噷浣跨敤 CAIL 閲囨牱闆嗭級銆?- `--law_dir ...`锛氭寚瀹氭硶鏉″簱鏂囦欢璺緞銆?- `--limit 100`锛氫粎璇勬祴鍓?100 鏉℃牱鏈€?- `--offset 0`锛氫粠绗?0 鏉℃牱鏈紑濮嬨€?- `--topk_law 3`锛氭绱?top-3 娉曟潯鍊欓€変緵妯″瀷閫夋嫨銆?- `--output_path ...jsonl`锛氭寚瀹氳瘎娴嬭緭鍑?JSONL 鏂囦欢璺緞銆?
## eval_llm_rag_penalty 指令

```bash
# CAIL sampled set
python eval_llm_rag_penalty.py --dataset_path data/testset/test_cail_sampled_single_seed_42.json --precedent_file data/candidates/precedents_cail.json --law_dir data/law_articles --limit 1000 --offset 0 --topk_law 3 --topk_case 3 --output_path embedding_output/llm_rag_penalty_cail.jsonl

# cjo22 full testset
python eval_llm_rag_penalty.py --dataset_path data/testset/testset.json --precedent_file data/candidates/precedent_case.json --law_dir data/law_articles --limit 1698 --offset 0 --topk_law 3 --topk_case 3 --output_path embedding_output/llm_rag_penalty_cjo22.jsonl
```

参数说明：
- `eval_llm_rag_penalty.py`：执行量刑预测的 LLM+RAG 基线评测。
- `--dataset_path ...`：指定待评测数据集路径。
- `--precedent_file ...`：指定相似案例候选文件路径。
- `--law_dir ...`：指定法条库目录路径。
- `--limit`：评测样本数上限，`0` 表示全量。
- `--offset`：从第几条样本开始评测。
- `--topk_law`：检索法条候选数量。
- `--topk_case`：检索相似案例候选数量。
- `--output_path ...jsonl`：指定评测输出 JSONL 文件路径。
## baseline plain_prompt 鎸囦护

```bash
python baseline/plain_prompt.py --task cail2018 --prompt-type plain_prompt --model qwen3-235b-a22b --dataset data/testset/test_cail_sampled_single_seed_42.json --limit 1000 --offset 0 --output-dir baseline_output/plain_prompt_cail2018
# cjo22 plain_prompt baseline (qwen3-max, full testset)
python baseline/plain_prompt.py --task cjo22 --prompt-type plain_prompt --model qwen3-max --dataset data/testset/testset.json --limit 1698 --offset 0
```

## calc_metrics 鎸囦护

```bash
python calc_metrics.py --input baseline_output\plain_prompt_cail2018_20260311_090456\plain_prompt_results_qwen3-235b-a22b.jsonl --mode ljp
```

