from tqdm import tqdm 

from random import uniform, randint, shuffle

from numpy import concatenate, stack
from sklearn.metrics import accuracy_score

from kymatio.torch import Scattering2D

from torch import Tensor, FloatTensor, no_grad, from_numpy
from torch import stack as tstack
from torch.fft import fft2, fftshift
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.nn import Module, Sequential, Linear, Tanh, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA

from src.data.base import BaseDataset
from src.data.attacks import *

from typing import Any





class DetectionDataset(Dataset): 


    def __init__(
            self, 
            dataset: BaseDataset,
            discriminant: str,
            discriminant_params: dict[str, Any],
            pca_n_components: int,
            n_imgs: int,
            device: str,
            is_text: bool = False
    ): 
        self.dataset = dataset
        self.discriminant = discriminant
        self.discriminant_params = discriminant_params
        self.num_samples = len(dataset)
        self.dataset_len = int(self.num_samples / 100) - 2
        self.cache = []
        self.device = device
        self.num_features = None
        self.n_imgs = n_imgs
        self.pca_n_components = pca_n_components
        self.is_text = is_text
        return 
    

    def __len__(self) -> int: 
        return self.dataset_len
    

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]: 
        return self.cache[index]
    

    def load(
            self
    ): 
        # Setup 
        self.dataset.load()
        indexes = list(range(self.num_samples))
        shuffle(indexes)

        # Create imgs
        imgs0, imgs1, imgs2, imgs3 = [], [], [], []
        for i in indexes: 
            x, _ = self.dataset[i]
            if not self.is_text: 
                if x.size(0) == 3:
                    x = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2] 
                elif x.size(0) == 1:
                    x = x[0]

            if randint(0, 1): 
                x = x.numpy()
                imgs0.append(x)

            else: 
                if self.is_text: 
                    attack = ShiftEmbedding(uniform(.4, .8))
                    imgs1.append(attack(x[None, :, :]).numpy())
                
                else: 
                    atk = randint(0, 2)
                    if atk == 0: 
                        attack = GaussianNoise(uniform(.5, 2.)) 
                        imgs1.append(attack(x[None, :, :])[0].numpy())
                    elif atk == 1:
                        attack = GaussianBlur(2 * randint(1, 9) + 1)
                        imgs2.append(attack(x[None, :, :])[0].numpy())
                    else: 
                        attack = RandomCancellation(uniform(.25, .75))
                        imgs3.append(attack(x[None, :, :])[0].numpy())

        cache_tmp = []
        labels_tmp = []

        for (imgs, label) in zip([imgs0, imgs1, imgs2, imgs3], [0, 1, 1, 1]): 
            num_imgs = len(imgs)
            num_clients = int(num_imgs / self.n_imgs)

            for i in range(num_clients):
                X = stack(imgs[i * self.n_imgs : (i + 1) * self.n_imgs])
                num_samples, height, width = X.shape

                pca = PCA(n_components = self.pca_n_components, random_state = 42).fit(X.reshape(num_samples, -1))
                eigimg = pca.explained_variance_ratio_.dot(pca.components_) / pca.explained_variance_ratio_.sum()
                eigimg = from_numpy(eigimg)
                eigimg = eigimg.view(height, width).type(FloatTensor)

                cache_tmp.append(eigimg)
                labels_tmp.append(label)

        # Cache 
        cache_tmp = tstack(cache_tmp)

        if self.discriminant == "fft": 
            beta = self.discriminant_params["beta"]

            h0, w0 = height // 2,  width // 2 
            H, W = int(beta * h0), int(beta * w0)

            cache = fft2(cache_tmp, dim = (-2, -1))
            cache = fftshift(cache).abs()
            cache = cache[:, h0-H : h0+H, w0-W : w0+W].reshape(cache.size(0), -1)

        elif self.discriminant == "wst": 
            J_wst = self.discriminant_params["J_wst"]
            L_wst = self.discriminant_params["L_wst"]

            wst = Scattering2D(J = J_wst, shape = (height, width), L = L_wst).to(self.device)
            cache = wst(cache_tmp.to(self.device))[:, 1:L_wst * J_wst].reshape(cache_tmp.size(0), -1).cpu()

        if self.num_features is None:
            self.num_features = cache.size(1)

        self.cache = [it for it in zip(cache, labels_tmp)]

        return 
    

    def clear(self):
        self.dataset.clear()
        self.cache = []
        return





class Detector(Module):


    def __init__(
            self,
            in_features: int,
    ):
        super(Detector, self).__init__()
        self.hl = Sequential(
            Linear(in_features, in_features // 2), Tanh(),
            Linear(in_features // 2, in_features // 2), Tanh()
        )
        self.clf = Linear(in_features // 2, 2)
        return 
    
    
    def forward(
            self, 
            X: Tensor
    ) -> Tensor: 
        out = (X - X.mean(1, keepdim = True)) / (X.std(1, keepdim = True) + 1e-8)
        out = self.hl(out)
        out = self.clf(out)
        return out





def train(
        dataloader: DataLoader,
        model: Module, 
        criterion: Module,
        optimizer: Optimizer,
        device: str
): 
    model.train()
    for X, y in dataloader: 
        X = X.to(device)
        y = y.to(device)
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return





@no_grad()
def evaluate(
        dataloader: DataLoader,
        model: Module, 
        device: str
) -> float: 
    model.eval()
    y_full = []
    y_hat_full = []
    for X, y in dataloader: 
        y_full.append(y.numpy())
        X = X.to(device)
        y = y.to(device)
        y_hat_full.append(model(X).cpu().argmax(1).numpy())
    y_full = concatenate(y_full, axis = 0)
    y_hat_full = concatenate(y_hat_full, axis = 0)
    return accuracy_score(y_full, y_hat_full)





def train_detector(
        train_dataset: BaseDataset,
        test_dataset: BaseDataset,
        discriminant: str,
        discriminant_params: dict[str, Any],
        num_epochs: int,
        batch_size: int, 
        pca_n_components: int,
        n_imgs: int = 100,
        device: str = "cpu",
        is_text: bool = False
) -> Module: 
    # Dataset
    acc_list = []
    train_detection_dataset = DetectionDataset(train_dataset, discriminant, discriminant_params, pca_n_components, n_imgs, device, is_text) 
    test_detection_dataset = DetectionDataset(test_dataset, discriminant, discriminant_params, pca_n_components, n_imgs, device, is_text) 

    train_dataloader = DataLoader(train_detection_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    test_dataloader = DataLoader(test_detection_dataset, batch_size = batch_size, shuffle = False, drop_last = False)

    # Load data + attacks 
    train_detection_dataset.load()
    test_detection_dataset.load()

    # Model 
    model = Detector(train_detection_dataset.num_features).to(device)
    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss()

    # Epochs
    for epoch in tqdm(range(num_epochs)): 
        # Training 
        train(train_dataloader, model, criterion, optimizer, device)

        # Evaluation 
        acc = evaluate(test_dataloader, model, device)
        acc_list.append(acc)
        print(f"- Epoch {epoch + 1}: accuracy_score = {round(acc, 4)}")

        # Reload attacks
        if epoch < num_epochs - 1:
            train_detection_dataset.clear()
            test_detection_dataset.clear()
            train_detection_dataset.load()
            test_detection_dataset.load()

    # Clear buffer
    train_detection_dataset.clear()
    test_detection_dataset.clear()

    return model, acc_list













