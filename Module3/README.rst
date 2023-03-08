# For Running

python src\ingest_data.py --input_path <path> --split_ratio <percentage> --processed_data <path>

python src\train.py --input_path <path> --weight_path <path>

python src\score.py --input_path <path> --weight_path <path>

For my case

python src\ingest_data.py --input_path data/raw/ --split_ratio 0.2 --processed_data data/processed

python src\train.py --input_path data/processed --weight_path artifacts 

python src\score.py --input_path data/processed --weight_path artifacts 