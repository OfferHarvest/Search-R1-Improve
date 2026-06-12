
index="<path-to-e5-flat-index>"

split -b 40G "$index" part_

python upload.py
