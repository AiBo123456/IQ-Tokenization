# TOkenized Time series EMbeddings for the CASE project
### 1. Setup your environment 
```
pip install -r requirements.txt
```
### 2. Create the folders /IQ_data and /IQ_normalized_data under the project root
### 3. Save .mat files inside the /IQ_data folder
### 4. Generate nomalized dataset
```
python convert_raw_data.py
```
### 5. Train VQVAE model
```
python train_vqvae.py
```
### 6. Check the Result
Find training curves online [here](https://www.comet.com/ai-bo/totem/view/new/panels).
