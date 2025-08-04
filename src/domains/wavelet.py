from random import shuffle

from torch import Tensor, stack, from_numpy
from kymatio.torch import Scattering2D

from sklearn.decomposition import PCA

from src.data.base import BaseDataset 




def get_dataset_domain(
        dataset: BaseDataset,
        J_wst: int = 2, 
        L_wst: int = 8,
        pca_n_components: int = 1, 
        sample_size: float = 1.,
        device: str = "cpu"
) -> dict[str, Tensor]: 
    # Loading
    N = len(dataset)
    num_samples = max(2, int(sample_size * N))

    dataset.load()
    shuffle(dataset.cache)
    imgs = [img for img, _ in dataset.cache[:num_samples]]
    dataset.clear()

    X = stack(imgs)
    X = 0.2989 * X[:, 0] + 0.5870 * X[:, 1] + 0.1140 * X[:, 2] if X.size(1) > 1 else X[:, 0]
    num_samples, height, width = X.size()

    # PCA 
    eigimg = []

    # PCA 
    pca = PCA(n_components = pca_n_components, random_state = 42).fit(X.numpy().reshape(num_samples, -1))
    eigimg = pca.explained_variance_ratio_.dot(pca.components_) / pca.explained_variance_ratio_.sum()
    eigimg = from_numpy(eigimg)
    eigimg = eigimg.view(1, height, width)

    # WST 
    wst = Scattering2D(J = J_wst, shape = (height, width), L = L_wst).to(device)
    eigwst = wst(eigimg.to(device))[:, 1 : J_wst * L_wst].reshape(-1).cpu()

    return eigwst









