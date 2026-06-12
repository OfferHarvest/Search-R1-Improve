
DATA_NAME=nq

DATASET_PATH="<path-to-qa-dataset>"

SPLIT='test'
TOPK=3

INDEX_PATH="<path-to-retrieval-index-directory>"
CORPUS_PATH="<path-to-wikipedia-corpus-jsonl>"
SAVE_NAME=e5_${TOPK}_wiki18.json

# INDEX_PATH="<path-to-alternative-index-directory>"
# CORPUS_PATH="<path-to-alternative-corpus-jsonl>"
# SAVE_NAME=e5_${TOPK}_wiki21.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python retrieval.py --retrieval_method e5 \
                    --retrieval_topk $TOPK \
                    --index_path $INDEX_PATH \
                    --corpus_path $CORPUS_PATH \
                    --dataset_path $DATASET_PATH \
                    --data_split $SPLIT \
                    --retrieval_model_path "intfloat/e5-base-v2" \
                    --retrieval_pooling_method "mean" \
                    --retrieval_batch_size 512 \
