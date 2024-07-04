import seaborn as sns
import data_utils
from sparse_autoencoder_KL import SparseAutoencoderKL
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# from simple_autoencoder import Autoencoder
# from sparse_autoencoder_l1 import SparseAutoencoderL1

if __name__ == '__main__':

    # autoencoder = Autoencoder()
    # sparse_autoencoder_l1 = SparseAutoencoderL1()
    sparse_autoencoder_kl = SparseAutoencoderKL()

    # autoencoder.load_state_dict(torch.load('./history/simple_autoencoder.pt'))
    # sparse_autoencoder_l1.load_state_dict(torch.load('./history/sparse_autoencoder_l1.pt'))
    sparse_autoencoder_kl.load_state_dict(torch.load(
        './history/sparse_autoencoder_KL.pt', map_location=torch.device('cpu')))

    # autoencoder.cpu()
    # sparse_autoencoder_l1.cpu()
    sparse_autoencoder_kl.cpu()

    BATCH_SIZE = 128
    train_loader, test_loader = data_utils.load_mnist(BATCH_SIZE)
    # Extract one image for each digit (0 through 9)
    digits = list(range(10))
    images = []
    labels = []
    for img, label in test_loader.dataset:
        if label in digits:
            images.append(img)
            labels.append(label)
            digits.remove(label)
        if not digits:
            break
    # Sort images and labels by label
    sorted_images = [img for _, img in sorted(
        zip(labels, images), key=lambda pair: pair[0])]
    sorted_labels = sorted(labels)

    images = torch.stack(sorted_images)
    labels = torch.tensor(sorted_labels)

    # Get the vector representation from the encoder
    with torch.no_grad():
        encoded_imgs = sparse_autoencoder_kl.encoder(images.view(-1, 784))

    """Vector Representation"""

    encoded_imgs_np = encoded_imgs.numpy()

    plt.figure(figsize=(12, 6))
    sns.heatmap(encoded_imgs_np, cmap='binary', cbar=True,
                linewidths=0.1, linecolor='black', yticklabels=sorted_labels)
    plt.title('Sparsity in Vector Representation (Latent Space) for Digits 0-9')
    plt.xlabel('Latent Dimensions')
    plt.ylabel('Digits')
    plt.savefig('./images/vector_representation_sparsity_digits.png')
    # plt.show()

    """Inner Product"""

    encoded_imgs_np = encoded_imgs.numpy()

    # Calculate inner product values between each pair of latent vectors
    inner_product_matrix = np.dot(encoded_imgs_np, encoded_imgs_np.T)

    plt.figure(figsize=(10, 8))
    sns.heatmap(inner_product_matrix, annot=True, fmt=".2f", cmap='binary',
                xticklabels=sorted_labels, yticklabels=sorted_labels)
    plt.title('Inner Product of Latent Vectors for Digits 0-9')
    plt.xlabel('Digits')
    plt.ylabel('Digits')
    plt.savefig('./images/inner_product_heatmap.png')
    # plt.show()
