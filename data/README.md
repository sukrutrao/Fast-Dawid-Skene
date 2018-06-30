## Input data

### Description

Input data must be provided in a CSV format. There are two types of inputs:
* Crowd Annotations
* Ground Truths

#### Crowd Annotations
This is a CSV file consisting of three columns - Annotator ID, Question ID, Annotation ID. The program assumes that all questions are annotated, and all possible annotations appear at least once.

#### Ground Truths
This is a CSV file consisting of two columns - Question ID, Annotation ID.

### Adding a new dataset
* Create a new directory in this directory. If the name of the dataset is 'mydataset', the directory name should be `mydataset_dataset`. To use datasets stored in a different path, use the `--dataset_path` flag while calling the script.
* The crowd annotations must be stored in a file inside the dataset directory as `crowd.csv`. The ground truths must be stored as `gold.csv`. For using crowd annotation and ground truth files from other paths, use the flags `--crowd_annotations_path` and `--ground_truths_path` respectively, when calling the script. If set, these override the default as well as the `--dataset_path` flag.

### Example
An example dataset, [`toy_dataset`](toy_dataset), is provided.