# Legal & Policy Document Segmentation

This repository provides a suite of tools to segment structured legal and policy documents using the OpenAI API. The process involves tagging individual lines of a document and then using those tags to create meaningful segments.

The data for this project is in data.zip, which contains the data for finetunning as well as the entire database.
The data is shared with git LFS, to retrieve the data
run
git lfs install
git clone *this repo*
git lfs pull

for more detailed instructions on git LFS, see:
https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage

---

## Prerequisites

Before you begin, ensure you have the following set up:

1.  **OpenAI Account:** You need an account with OpenAI.
2.  **API Key:** You must generate an API key from your OpenAI account dashboard.
3.  **Environment Variable:** For security, it's best to set your API key as an environment variable. This project expects the key to be available as `OPENAI_API_KEY`.
4.  **Python Dependencies:** Install the necessary Python packages. It's recommended to create a `requirements.txt` file and install from it using:
    pip install -r requirements.txt

---

## Usage Workflow

Follow these steps to process your documents from raw text to a segmented CSV file.

### 1. Prepare Your Data

Place your raw text files in the `sample_data/txt_files/` directory. **Crucially, each line in every file must be prefixed with a line number**, followed by a space, as shown in the provided examples.

### 2. Submit Tagging Jobs

Run the batch_itemizer.py script to submit your text files to the OpenAI API for tagging. This script creates batch jobs for asynchronous processing.

python batch_itemizer.py

### 3. Retrieve Batch Results
After you've confirmed that all submitted batch jobs have been completed in the OpenAI dashboard, run the retrieve_batches.py script. This will fetch the processing results (line-by-line tags).

python retrieve_batches.py

### 4. Reconstruct Tagged Files
Execute the reconstruct_label_to_txt.py script to assemble the retrieved tags with the original text. The newly tagged files will be saved in the sample_data/txt_files_tagged/ directory.

python reconstruct_label_to_txt.py

### 5. Segment the Documents
Run the segmentation.py script to divide the tagged text files into logical segments. You may need to adjust the parameters within this script until you achieve a satisfactory level of segmentation.

python segmentation.py

### 6. Create Final CSV Output
Finally, run create_table_from_segmented_files.py to organize the segmented documents into a single .csv file. This file can then be used for further data analysis and processing.

python create_table_from_segmented_files.py

