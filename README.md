# AIM Lab Session

## Environment setup
```bash
pip install -r requirements.txt
mkdir model_dir
```

## Getting datasets
### Metmusem 
```bash
mkdir -p data/metmuseum
wget 'https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv' -O data/metmuseum/MetObjects.csv
```
### MNIST
1. Download the [MNIST as \.jpg dataset](https://www.kaggle.com/datasets/scolianni/mnistasjpg) (Kaggle account required) - saved as `archive.zip`
2. Extract to data dir:
```bash
mkdir -p data/MNIST/raw
unzip archive.zip -d data/MNIST/raw
mv data/MNIST/raw/trainingSet/trainingSet/ data/MNIST/raw
rm -r data/MNIST/testS* data/MNIST/trainingS* # remove unneeded files
```
3. Create filepath to label mapping:
```
cd data/MNIST && chmod +x make_csv.sh
./make_csv.sh raw
```

