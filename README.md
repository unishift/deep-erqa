# Deep ERQA

NN-based Full-Reference metric for assessing image restoration quality.

## Usage example

0. Install requirements `pip install -r requirements.txt`
1. Download weights: https://drive.google.com/file/d/11oo-EoJN5yVQ30m0PwBA4qgqpY3NBg3m/view?usp=sharing
2. Unzip downloaded file
3. Modify `edge_similarity/test_benchmark.py` for your needs and execute.
```
python edge_similarity/test_benchmark.py --gpu 0 ${MODEL_PATH}/best_model/model ${DATA_PATH} ${RES_PATH}
```

## Train example

0. Install requirements `pip install -r requirements.txt`
1. Download `fannet.zip` from [this link](https://drive.google.com/drive/folders/1dOl4_yk2x-LTHwgKBykxHQpmqDvqlkab) and unzip to `${DATA_PATH}`
2. Run the following command
```
python edge_similarity/train.py --gpu 0 ${DATA_PATH}
```