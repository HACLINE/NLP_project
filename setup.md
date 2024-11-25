# Setup Instructions

## Step 1: Create a conda environment & install dependencies
```bash
conda create -n scrl python=3.8
conda activate scrl
pip install -r requirements.txt
pip install -e .
```

## Step 2: Download data
The extra terms can be found in [this tsinghua cloud folder](https://cloud.tsinghua.edu.cn/d/bbea8404be7b4726a472/). After downloading, execute the following commands
```bash
unzip data.zip
unzip all-distilroberta-v1.zip
unzip distilroberta-base.zip
unzip punkt.zip
cp -r punkt ~/nltk_data/tokenizers
```