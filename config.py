from pathlib import Path

def get_config():
    return {
        "batch_size": 4, 
        "num_epochs": 2,
        "lr": 10 ** -4,
        "seq": 350,
        "d_model": 512,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "exp_name": "runs/tmodel"
    }

def get_weights_file_path(config, epochs: str):
    model_folder = f"{config["datasource"]}_{config["model_folder"]}"
    model_filename = f"{config["model_basename"]}{epochs}.pt"
    return str(Path(".")/model_folder/model_filename)

def latest_weights(config):
    model_folder = f"{config["datasource"]}_{config["model_folder"]}"
    model_filename = f"{config["model_basename"]}*"

    weights_files = list(Path(model_folder).glob(model_filename))

    if len(weights_files) == 0:
        return None

    weights_files.sort()
    return str(weights_files[-1])



