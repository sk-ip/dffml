dffml merge text=df temp=csv \
    -source-text-dataflow preprocess_ops.json \
    -source-text-features city:str:1 state:str:1 month:int:1 \
    -source-text-source csv \
    -source-text-source-filename dataset.csv \
    -source-temp-filename preprocessed.csv \
    -source-temp-allowempty \
    -source-temp-readwrite \
    -log debug