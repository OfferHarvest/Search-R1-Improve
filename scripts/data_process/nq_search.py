# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess QA datasets to Search-R1 parquet format.

Examples:
    python scripts/data_process/nq_search.py --local_dir ./data/nq_search
    python scripts/data_process/nq_search.py --local_dir ./data/nq_hotpotqa_train --data_sources nq,hotpotqa
    python scripts/data_process/nq_search.py --local_dir ./data/nq_hotpotqa_train \
        --train_data_sources nq,hotpotqa \
        --test_data_sources nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_sources', default='nq',
                        help='Comma-separated data sources used for both train and test.')
    parser.add_argument('--train_data_sources', default=None,
                        help='Comma-separated training data sources. Overrides --data_sources for train.')
    parser.add_argument('--test_data_sources', default=None,
                        help='Comma-separated test data sources. Overrides --data_sources for test.')
    parser.add_argument('--cache_dir', default=None,
                        help='Optional HuggingFace datasets cache dir.')

    args = parser.parse_args()

    def parse_sources(value):
        return [source.strip() for source in value.split(',') if source.strip()]

    train_sources = parse_sources(args.train_data_sources or args.data_sources)
    test_sources = parse_sources(args.test_data_sources or args.data_sources)

    def load_flashrag_dataset(data_source):
        load_kwargs = {}
        if args.cache_dir is not None:
            load_kwargs['cache_dir'] = os.path.join(args.cache_dir, f'{data_source}_cache')
        return datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source, **load_kwargs)

    def make_map_fn(data_source, split):
        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    def select_split(dataset, data_source, preferred_split):
        if preferred_split in dataset:
            print(f'Using the {data_source} {preferred_split} dataset...')
            return dataset[preferred_split], preferred_split
        if preferred_split == 'test' and 'dev' in dataset:
            print(f'Using the {data_source} dev dataset as test...')
            return dataset['dev'], 'dev'
        if 'train' in dataset:
            print(f'Using the {data_source} train dataset as {preferred_split}...')
            return dataset['train'], 'train'
        raise ValueError(f'No usable split found for {data_source}: {list(dataset.keys())}')

    all_train_datasets = []
    for data_source in train_sources:
        dataset = load_flashrag_dataset(data_source)
        train_dataset, split_name = select_split(dataset, data_source, 'train')
        train_dataset = train_dataset.map(
            function=make_map_fn(data_source, split_name),
            with_indices=True
        )
        all_train_datasets.append(train_dataset)

    all_test_datasets = []
    for data_source in test_sources:
        dataset = load_flashrag_dataset(data_source)
        test_dataset, split_name = select_split(dataset, data_source, 'test')
        test_dataset = test_dataset.map(
            function=make_map_fn(data_source, split_name),
            with_indices=True
        )
        all_test_datasets.append(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset = datasets.concatenate_datasets(all_train_datasets)
    test_dataset = datasets.concatenate_datasets(all_test_datasets)

    train_path = os.path.join(local_dir, 'train.parquet')
    test_path = os.path.join(local_dir, 'test.parquet')
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print(f'Saved train dataset ({len(train_dataset)} rows) to {train_path}')
    print(f'Saved test dataset ({len(test_dataset)} rows) to {test_path}')

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
