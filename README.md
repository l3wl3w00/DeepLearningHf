# Deep Learning Class (VITMMA19) Project Work template


## Submission Instructions
### Data Preparation

**Important:** You must provide a script (or at least a precise description) of how to convert the raw database into a format that can be processed by the scripts.
* The scripts should ideally download the data from there or process it directly from the current sharepoint location.
* Or if you do partly manual preparation, then it is recommended to upload the prepared data format to a shared folder and access from there.

### Data Preparation

- Discovers images under --input that match the naming convention.
- Loads all JSON exports under --input (recursively).
- Extracts one label per task/entry and annotator.
- Normalizes filenames so JSON `file_upload` keys match disk basenames.
- Normalizes labels into exactly 3 classes: Pronation / Neutral / Supination.
- Computes majority label per image across annotators.
- Exports:
    - image_manifest.csv
    - all_annotations.csv
    - majority_labels.csv
    - label_summary.json

Before submitting your project, ensure you have completed the following steps.
**Please note that the submission can only be accepted if these minimum requirements are met.**

- [ ] **Project Information**: Filled out the "Project Information" section (Topic, Name, Extra Credit).
- [ ] **Solution Description**: Provided a clear description of your solution, model, and methodology.
- [ ] **Extra Credit**: If aiming for +1 mark, filled out the justification section.
- [ ] **Data Preparation**: Included a script or precise description for data preparation.
- [ ] **Dependencies**: Updated `requirements.txt` with all necessary packages and specific versions.
- [ ] **Configuration**: Used `src/config.py` for hyperparameters and paths, contains at least the number of epochs configuration variable.
- [ ] **Logging**:
    - [ ] Log uploaded to `log/run.log`
    - [ ] Log contains: Hyperparameters, Data preparation and loading confirmation, Model architecture, Training metrics (loss/acc per epoch), Validation metrics, Final evaluation results, Inference results.
- [ ] **Docker**:
    - [ ] `Dockerfile` is adapted to your project needs.
    - [ ] Image builds successfully (`docker build -t dl-project .`).
    - [ ] Container runs successfully with data mounted (`docker run ...`).
    - [ ] The container executes the full pipeline (preprocessing, training, evaluation).
- [ ] **Cleanup**:
    - [ ] Removed unused files.
    - [ ] **Deleted this "Submission Instructions" section from the README.**

## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Györgydeák Levente
- **Aiming for +1 Mark**: No

### Solution Description

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
