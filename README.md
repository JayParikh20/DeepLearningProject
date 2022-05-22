# Self Drviing Car Deep Learning Project

## Installation Setup
Install pre-requisites of the simulator and other depended packaged using 
> pip install -r requirements.txt

## For Training
1. Donwload the DATA_1.zip model using this link [Google Drive Link](https://drive.google.com/drive/folders/1AViLJLD-hH5JD4UnCZ7pi4lqX8ZQdx4m?usp=sharing)
2. Upload the Network_trainer_modified.ipynb jupyter notebook to Google Colab
3. Copy DATA_1.zip to google colab files section
4. Run all cells
5. Trained model will be automatically installed.

## For Testing
1. Install the latest version of CARLA (tested on 0.9.13). 
2. Change the directory to CARLA and run carla simualtor using
> ./CarlaUE4.sh or CarlaUE4.exe
3. Copy the trained model to models folder (1.h5)
4. (optional) change model path name in ud_car_run.py using MODEL_PATH if needed
5. run python script ud_car_run.py to test the model.
