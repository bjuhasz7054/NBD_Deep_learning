# NBD FairFace Project

Classifier CNN for recognizing facial features from the fairface dataset and latent space examination.
If you want to know more about the project read the [project description](project_description.md).
Members of the team: Domonkos Debreczeni - Q69B8U, Benedek Juhász - A2PMXC, Nándor Szécsényi - RJ448X

## Docker
```bash
docker build .
docker run -it <image hash>
```
If you want to use gpus, then [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) is needed
```bash
docker run --gpus all -it <image hash>
```
Mounting volumes for the dataset and model checkpoint / evaluation results is a good idea
```bash
docker run --gpus all -v dataset:/usr/src/dataset -v results:/usr/src/results <image hash>
```
If you want to edit source code without having to rebuild image mount src folder as well

## Running
Have the folder root in your PYTHONPATH (not needed in docker)
```bash
export PYTHONPATH=$(pwd)
```


#### Config
You can tune configuration in `config.ini`.
Batch sizes might need to be decreased depending on the availible memory.

#### Collect Dataset
```bash
python bin/collect_dataset.py
```
```
usage: collect_dataset.py [-h] [-d DATASET_FOLDER] [-o] [-k]

Collect FairFace dataset

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET_FOLDER, --dataset_folder DATASET_FOLDER
  -o, --overwrite       overwrite existing dataset
  -k, --keep_zip        do not remove zip file after extracting
```

#### Training and Evaluating
```bash
python bin/run.py train
python bin/run.py evaluate -l <model file>
```
```
usage: run.py [-h] [-d DATASET_FOLDER] [-r RESULTS_FOLDER] [-l LOAD_MODEL] {train,evaluate}

train or evaluate Face classification model on FairFace dataset

positional arguments:
  {train,evaluate}

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET_FOLDER, --dataset_folder DATASET_FOLDER
  -r RESULTS_FOLDER, --results_folder RESULTS_FOLDER
                        Folder to save model checkpoints or save evaluations
  -l LOAD_MODEL, --load_model LOAD_MODEL
                        Load model to continue training or to evaluate it
```

## Dependency Management (dev)
Setup virtual environment
```bash
pip install pipenv
pipenv install --dev
```
Enter virtual environment
```bash
pipenv shell
# or run single command
pipenv run <cmd>
```
Update after adding new package to Pipfile
```bash
pipenv update --dev
```
## Saved model
A saved model file can be found and downloaded from here: https://drive.google.com/file/d/1kDAue82VnDygPI_YwHV2TKRR8XRJGcaj/view?usp=sharing
## Other files
There are two extra Jupyter notebooks, these are meant for visualization and other extras, such as showing the latent vector space analysis. Running these notebooks is fairly simple, just use the usual notebook controls or the "Select all" command in the Runtime menu (don't forget to change the runtime type to GPU). Also before running you need to download the model file and testing pictures(media folder).
