from random import shuffle 

from torch import Tensor, stack, from_numpy
from torch.fft import fft2, fftshift

from sklearn.decomposition import PCA

from src.data.base import BaseDataset





def get_dataset_domain(
        dataset: BaseDataset,
        beta: float = 0.1,
        pca_n_components: int = 1, 
        sample_size: float = 1.
) -> Tensor: 
    # Loading 
    N = len(dataset)
    num_samples = max(2, int(sample_size * N))

    dataset.load()
    shuffle(dataset.cache)
    imgs = [img for img, _ in dataset.cache[:num_samples]]
    dataset.clear()

    X = stack(imgs)
    if X.size(1) == 3:
        X = 0.2989 * X[:, 0] + 0.5870 * X[:, 1] + 0.1140 * X[:, 2] 
    elif X.size(1) == 1:
        X = X[:, 0]
    num_samples, height, width = X.size()

    # PCA 
    pca = PCA(n_components = pca_n_components, random_state = 42).fit(X.numpy().reshape(num_samples, -1))
    eigimg = pca.explained_variance_ratio_.dot(pca.components_) / pca.explained_variance_ratio_.sum()
    eigimg = from_numpy(eigimg)
    eigimg = eigimg.view(1, height, width)

    # Fourier 
    h0, w0 = height // 2,  width // 2 
    H, W = int(beta * h0), int(beta * w0)

    eigfft = fft2(eigimg, dim = (-2, -1))
    eigfft = fftshift(eigfft).abs()
    eigfft = eigfft[:, h0-H : h0+H, w0-W : w0+W]
    eigfft = eigfft.reshape(1, -1)

    return eigfft


