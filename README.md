# TOkenized Time series EMbeddings for the CASE project
### 1. Setup your environment 
```
pip install -r requirements.txt
```
### 2. Create the folders /IQ_data and /IQ_normalized under the project root. Save .mat files inside the /IQ_data folder.
### 3. Generate nomalized dataset
```
python convert_raw_data.py
```
### 4. Train VQVAE model
```
python train_vqvae.py
```
### 5. Check the Result
Find training curves online [here](https://www.comet.com/ai-bo/totem/view/new/panels).
