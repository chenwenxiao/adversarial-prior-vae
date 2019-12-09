from . import loss

def plot_vae_loss_curves():
    kw_list = [
        "VAE D loss",
        "VAE D real",
        "VAE G loss",
        "batch VAE loss",
        "train grad penalty",
    ]
    # setting this path for usage in train script
    loss.curves(kw_list=kw_list,src_path='../console.log',dst_path='../plotting/loss/')