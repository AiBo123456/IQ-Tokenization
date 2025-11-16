# TOkenized Time series EMbeddings for the CASE project
### 1. Setup your environment 
```
pip install -r requirements.txt
```
### 2. Generate nomalized dataset
```
python convert_raw_data.py
```
### 3. Train VQVAE model
```
python train_vqvae.py
```
### 4. Check the Result
Find training curve on comet websit [here]([https://www.comet.com/ai-bo/totem/view/new/panels]).
