from pathlib import Path

def get_config():
    return {
        "batch_size": 4, 
        "num_epochs": 20,
        "lr": 10 ** -4,
        "seq": 350,
        "d_model": 512,
        "datasource": "",
        "lang_src": "it",
        "model_folder": "",
        "model_basename": "",
        "preload": "latest",
        "tokenizer_file": "",
        "exp_name": ""
    }