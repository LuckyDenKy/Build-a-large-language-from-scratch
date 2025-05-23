from gpt_download import download_and_load_gpt2
from ch4 import GPTModel
from ch5 import load_weights_into_gpt

def prepare_model():
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length":1024,
        "drop_rate":0.0,
        "qkv_bias":True
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim":768,"n_layers":12,"n_heads":12},
        "gpt2-medium (355M)": {"emb_dim":1024,"n_layers":24,"n_heads":16},
        "gpt2-large (774M)": {"emb_dim":1280,"n_layers":36,"n_heads":20},
        "gpt2-xl (1558M)": {"emb_dim":1600,"n_layers":48,"n_heads":25},
    }
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings,params = download_and_load_gpt2(model_size=model_size,models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model,params)
    model.eval()
    return model


if __name__ == "__main__":
    model = prepare_model()
    print(model)
