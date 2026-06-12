"""
测试脚本：仿照 rollout 阶段向检索服务发送请求并打印返回结果。

用法:
    python test_retrieval.py --url http://127.0.0.1:8000/retrieve --topk 3 --query "What is machine learning?"
    python test_retrieval.py --url http://127.0.0.1:8000/retrieve --topk 5 --queries "query1" "query2" "query3"
    python test_retrieval.py  # 使用默认配置
"""

import argparse
import json
import requests


def batch_search(url: str, queries: list, topk: int, return_scores: bool = True) -> dict:
    """仿照 generation.py:_batch_search 发送 POST 请求"""
    payload = {
        "queries": queries,
        "topk": topk,
        "return_scores": return_scores,
    }

    print(f"[REQUEST] URL: {url}")
    print(f"[REQUEST] Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    print("-" * 60)

    resp = requests.post(url, json=payload)
    print(f"[DEBUG] search_url={url} status={resp.status_code} body={resp.text[:500]}", flush=True)

    if resp.status_code != 200:
        print(f"[ERROR] HTTP {resp.status_code}: {resp.text}")
        resp.raise_for_status()

    return resp.json()


def passages2string(retrieval_result: list) -> str:
    """仿照 generation.py:_passages2string 格式化检索结果"""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        score = doc_item.get("score", "N/A")
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        format_reference += f"  [score: {score:.4f}]\n" if isinstance(score, float) else f"  [score: {score}]\n"
    return format_reference


def main():
    parser = argparse.ArgumentParser(description="测试检索服务请求")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/retrieve",
                        help="检索服务地址 (默认: http://127.0.0.1:8000/retrieve)")
    parser.add_argument("--topk", type=int, default=3,
                        help="返回文档数量 (默认: 3)")
    parser.add_argument("--query", type=str, default=None,
                        help="单个查询字符串")
    parser.add_argument("--queries", type=str, nargs="+", default=None,
                        help="多个查询字符串")
    parser.add_argument("--no-scores", action="store_true",
                        help="不返回相关性分数")

    args = parser.parse_args()

    # 处理查询参数
    if args.queries:
        queries = args.queries
    elif args.query:
        queries = [args.query]
    else:
        # 默认测试查询
        queries = ["What is machine learning?"]

    # 发送请求
    resp_json = batch_search(
        url=args.url,
        queries=queries,
        topk=args.topk,
        return_scores=not args.no_scores,
    )

    # 打印原始响应
    print("\n" + "=" * 60)
    print("[RAW RESPONSE]")
    print(json.dumps(resp_json, ensure_ascii=False, indent=2)[:2000])
    print("=" * 60)

    # 仿照 rollout 阶段格式化打印每个 query 的检索结果
    results = resp_json.get("result", [])
    print(f"\n共 {len(results)} 个查询的检索结果:\n")

    for i, (query, passages) in enumerate(zip(queries, results)):
        print(f"{'='*60}")
        print(f"[Query {i+1}]: {query}")
        print(f"{'='*60}")
        formatted = passages2string(passages)
        print(formatted)


if __name__ == "__main__":
    main()
