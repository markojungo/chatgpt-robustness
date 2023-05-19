# ChatGPT Robustness with TextFlint

Note: use `nlp` conda environment: `conda activate /home/jupyter/conda_env/nlp`.

Make sure to run `pip install openai textflint` and add OpenAI secret key to .env file.

Downloading data is done in `download_data.ipynb`. Data transformations are done in `transform_data.ipynb`. Evaluation is done in `textflint.ipynb`. 

Downloaded data resides in `./data`. Transformed outputs are found in `./transformed_data/{TASK_NAME}`.

