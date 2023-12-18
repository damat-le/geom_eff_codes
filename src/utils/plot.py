import random
import numpy as np
import matplotlib.pyplot as plt

def plot_random_sample(dataset, model, model_type="conv", idx=None):
    """
    model_type: 'conv' or 'fully_connected'
    """
    if idx is None:
        idx = random.choice(range(len(dataset)-15000, len(dataset)))

    in_ = dataset[idx:idx+1][0]
    labels = dataset[idx:idx+1][1]

    if model_type == 'conv':
        in_rec = model.generate(in_, labels=labels).squeeze().detach().numpy() # for BVAE
    elif model_type == 'fully_connected':
        in_rec = model.generate(in_.reshape(-1,169)).reshape(13,13).detach().numpy() # for BVAE_MLP

    # use this with mnist
    # in_ = d[idx][0]
    # in_rec = model.generate(in_.reshape(1,*in_.shape)).squeeze().detach().numpy()

    in_rec_thrd = np.where(in_rec > 0.5, 1, 0)
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(in_.squeeze().detach().numpy(), cmap='binary')
    axs[0].set_title('Sample')

    axs[1].imshow(in_rec, cmap='binary')
    axs[1].set_title('Reconstruction')

    for ax in axs:
        ax.set_yticks([])
        ax.set_xticks(range(0,13,2))

    fig.show()
    
    print(f'Sample idx :  {idx}')
    print(f'Sample labels :  {labels}')

    return fig, axs