# AIM Lab Session

## Environment setup
```bash
pip install -r requirements.txt
mkdir model_dir
```

## Getting datasets
### Metmusem 
```bash
mkdir -p data/metmusem
wget 'https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv' -O data/metmuseum/MetObjects.csv
```
### MNIST
1. Download the [MNIST as \.jpg dataset](https://www.kaggle.com/datasets/scolianni/mnistasjpg) (Kaggle account required) - saved as `archive.zip`
2. Extract to data dir:
```bash
mkdir -p data/MNIST/raw
unzip archive.zip -d data/MNIST/raw
rm testSet.tar.gz trainingSet.tar.gz  #  Remove intermediate files
mv data/MNIST/trainingSet/trainingSet/ data/MNIST/raw
rm -r data/MNIST/testS* data/MNIST/trainingS*
```
3. Create filepath to label mapping:
```
cd data/MNIST && chmod +x make_csv.sh
./make_csv.sh raw
```

