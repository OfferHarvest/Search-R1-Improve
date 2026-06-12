import argparse
import json
import os
import re
import string

import requests
import torch
import transformers


DEFAULT_MODEL_ID = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
DATASETS = {
    "bamboogle": "<path-to-bamboogle-test-jsonl>",
    # Add more datasets here, for example:
    # "nq": "<path-to-nq-test-jsonl>",
    # "hotpotqa": "<path-to-hotpotqa-test-jsonl>",
}
DEFAULT_SUMMARY_OUTPUT_PATH = "em_scores.json"
CURR_EOS = [151645, 151643]  # for Qwen2.5 series models
SEARCH_TEMPLATE = "\n\n{output_text}<information>{search_results}</information>\n\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Search-R1 on a jsonl QA dataset.")
    parser.add_argument(
        "--data_path",
        default=None,
        help="Optional single dataset path. If unset, infer.py evaluates DATASETS configured in code.",
    )
    parser.add_argument("--data_name", default="custom", help="Dataset name used with --data_path.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--search_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_searches", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples.")
    parser.add_argument(
        "--summary_output_path",
        default=DEFAULT_SUMMARY_OUTPUT_PATH,
        help="File that stores EM scores for each dataset.",
    )
    parser.add_argument("--output_path", default=None, help="Optional jsonl file for per-sample results.")
    return parser.parse_args()


def normalize_answer(text):
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s):
        return " ".join(s.split())

    def remove_punc(s):
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)
    return int(any(normalize_answer(answer) == normalized_prediction for answer in golden_answers))


def extract_answer(text):
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if not matches:
        return ""
    return matches[-1].strip()


def get_query(text):
    matches = re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def load_jsonl(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_prompt(question):
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"

    return f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
"""


def search(query, search_url, topk):
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True,
    }
    response = requests.post(search_url, json=payload, timeout=60)
    response.raise_for_status()
    results = response.json()["result"]

    formatted = ""
    for idx, doc_item in enumerate(results[0]):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        formatted += f"Doc {idx + 1}(Title: {title}) {text}\n"
    return formatted


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False)
            for target_sequence in target_sequences
        ]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i] :], target):
                return True

        return False


def run_one_sample(
    question,
    tokenizer,
    model,
    device,
    stopping_criteria,
    args,
):
    prompt = build_prompt(question)
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

    full_text = prompt
    num_searches = 0

    while True:
        input_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
        )

        generated_tokens = outputs[0][input_ids.shape[1] :]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_candidate = full_text + output_text
        answer = extract_answer(output_text)

        if outputs[0][-1].item() in CURR_EOS or answer:
            return {
                "prediction": answer,
                "full_output": full_candidate,
                "num_searches": num_searches,
            }

        if num_searches >= args.max_searches:
            return {
                "prediction": answer,
                "full_output": full_candidate,
                "num_searches": num_searches,
                "stopped_by": "max_searches",
            }

        query = get_query(full_candidate)
        search_results = search(query, args.search_url, args.topk) if query else ""
        full_text += SEARCH_TEMPLATE.format(output_text=output_text, search_results=search_results)
        num_searches += 1


def evaluate_dataset(
    dataset_name,
    data_path,
    tokenizer,
    model,
    device,
    stopping_criteria,
    args,
    output_file=None,
):
    total = 0
    correct = 0

    print(f"\n################# [Dataset: {dataset_name}] ##################\n")
    print(f"Data path: {data_path}")

    for sample in load_jsonl(data_path, args.limit):
        sample_id = sample.get("id", str(total))
        question = sample["question"]
        golden_answers = sample["golden_answers"]

        result = run_one_sample(
            question=question,
            tokenizer=tokenizer,
            model=model,
            device=device,
            stopping_criteria=stopping_criteria,
            args=args,
        )
        prediction = result["prediction"]
        score = em_check(prediction, golden_answers)

        total += 1
        correct += score
        running_em = correct / total

        print(
            f"[{dataset_name} {total}] id={sample_id} em={score} running_em={running_em:.4f} "
            f"prediction={prediction!r} golden={golden_answers}"
        )

        if output_file:
            output_file.write(
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "id": sample_id,
                        "question": question,
                        "golden_answers": golden_answers,
                        "prediction": prediction,
                        "em": score,
                        "num_searches": result["num_searches"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_file.flush()

    em = correct / total if total else 0.0
    print(f"\nDataset {dataset_name} EM: {em:.4f} ({correct}/{total})")
    return {
        "dataset": dataset_name,
        "data_path": data_path,
        "em": em,
        "correct": correct,
        "total": total,
    }


def main():
    args = parse_args()
    datasets_to_eval = (
        {args.data_name: args.data_path}
        if args.data_path
        else DATASETS
    )
    datasets_to_eval = {
        name: os.path.abspath(path)
        for name, path in datasets_to_eval.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    target_sequences = [
        "</search>",
        " </search>",
        "</search>\n",
        " </search>\n",
        "</search>\n\n",
        " </search>\n\n",
    ]
    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(target_sequences, tokenizer)]
    )

    output_file = None
    if args.output_path:
        output_file = open(args.output_path, "w", encoding="utf-8")

    summary = []

    try:
        for dataset_name, data_path in datasets_to_eval.items():
            summary.append(
                evaluate_dataset(
                    dataset_name=dataset_name,
                    data_path=data_path,
                tokenizer=tokenizer,
                model=model,
                device=device,
                stopping_criteria=stopping_criteria,
                args=args,
                    output_file=output_file,
                )
            )
    finally:
        if output_file:
            output_file.close()

    with open(args.summary_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n################# [Final EM Summary] ##################\n")
    for item in summary:
        print(f"{item['dataset']}: EM={item['em']:.4f} ({item['correct']}/{item['total']})")
    print(f"\nSaved EM summary to: {args.summary_output_path}")


if __name__ == "__main__":
    main()
