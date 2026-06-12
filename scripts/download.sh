
save_path="<path-to-retrieval-data>"

python download.py --save_path "$save_path"

cat "$save_path"/part_* > "$save_path"/e5_Flat.index
