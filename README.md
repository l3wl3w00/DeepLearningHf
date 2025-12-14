# Deep Learning Class (VITMMA19) Project Work template
## Data Preparation

The container expects all data under `/data`. On your host machine, you mount your dataset folder to `/data` inside the container.

Your local dataset folder should look like this:
```
your_data_root/
  <folder_1>/
    *.jpg|*.png|*.jpeg|*.bmp|*.tif|*.tiff
    *.json
  <folder_2>/
    *.jpg|*.png|*.jpeg|*.bmp|*.tif|*.tiff
    *.json
  ...
  consensus/
    *.json
```

Only images following this naming pattern are used: `source_identifier_sequence.ext`

This is the exact format in which files can be downloaded from the teams post about the easy way to download datasets

## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Györgydeák Levente
- **Aiming for +1 Mark**: No

### Solution Description
**Data preprocessing:**
A preprocessing script scans `/data` recursively, keeps only images matching the required filename
pattern `(sajat|internet_identifier_XX.ext)`, and writes a complete image manifest to `/app/output/image_manifest.csv.` 
It then loads every JSON export it finds (including consensus labels), normalizes filenames, normalizes label variants 
into exactly three classes, and saves:

- `/app/output/all_annotations.csv` (raw extracted labels)
- `/app/output/majority_labels.csv` (per-image majority vote)
- `/app/output/label_summary.json` (coverage + class distribution)

**Model architecture:**

Training uses EfficientNet-B0 pretrained on ImageNet, with the final classifier layer replaced
to output 3 logits (one per class). Images are resized to 224×224 and normalized using standard
ImageNet statistics.

**Training methodology:**

The labeled data is split into train/validation with a stratified 80/20 split to preserve 
class ratios. Optimization uses Adam with a learning rate of 0.02, early stopping (patience 10), 
and ReduceLROnPlateau to lower the learning rate when validation loss stagnates.

Instead of standard hard label cross entropy, the project uses a distance aware soft target
KL loss. A small cost matrix defines that mixing up Neutral with Pronation or Neutral with
Supination should be penalized less than mixing up Pronation with Supination. For each true 
label, a soft target probability distribution is built with q proportional to exp(-cost/temperature).
Training then minimizes KL divergence between this soft target distribution and the model prediction softmax. 

This makes the model treat near miss mistakes as less severe than clearly wrong predictions.

**Results:**

On the evaluation set, a simple baseline that always predicts the most frequent class (Neutral) achieves 52.47% accuracy. The trained EfficientNet-B0 model reaches 75.93% accuracy, a substantial improvement over the baseline. Training stopped early at epoch 24, with learning rate reductions helping stabilize validation performance.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.
[Adjust the commands that show how do build your container and run it with log output.]

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-hw .
```

#### Run

To run the full pipeline, start the container with your dataset mounted to /data, the outputs mounted to /app/output, and the log folder mounted to /app/log so logs are written directly to your host machine.
```bash
docker run --rm -it --gpus all -v ${PWD}/src:/app/src -v ${PWD}/notebook:/app/notebook -v /absolute/path/to/your/local/data:/data -v /absolute/path/to/your/local/output:/app/output -v /absolute/path/to/your/local/log:/app/log dl-hw /app/run.sh
```
Replace `/absolute/path/to/your/local/data` with the path to your dataset on your machine.

Replace `/absolute/path/to/your/local/output` with the directory where processed files and the trained model should be saved.

Replace `/absolute/path/to/your/local/log` with the directory where logs should be written. Since logging happens inside `/app/log`, the log files will be available directly on your host.

This command runs every step in order: data preprocessing, training, evaluation, and inference.

### File Structure and Functions
The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
