# AIM Lab Session

## Environment setup
```console
pip install -r requirements.txt
mkdir model_dir saved_models
```

## Getting datasets
### Metmusem 
Actually, use this to host the file online: https://www.station307.com/#/
```console
mkdir -p data/metmuseum
wget <station307-url> -O data/metmuseum/MetObjects.csv
```
### MNIST
1. Download the [MNIST as \.jpg dataset](https://www.kaggle.com/datasets/scolianni/mnistasjpg) (Kaggle account required) - will be saved as `archive.zip`
2. Extract to data dir:
```console
mkdir -p data/MNIST/raw
unzip archive.zip -d data/MNIST
mv data/MNIST/trainingSet/trainingSet/* data/MNIST/raw
rm -r archive.zip data/MNIST/testS* data/MNIST/trainingS* # remove unneeded files
```
3. Create filepath to label mapping:
```console
cd data/MNIST && chmod +x create_csv.sh
./create_csv.sh raw
```

