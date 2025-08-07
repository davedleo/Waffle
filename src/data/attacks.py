from torch import Tensor, rand, rand_like, zeros_like, normal, no_grad
from torchvision.transforms.functional import gaussian_blur





class Attack(object): 


    def __init__(
            self
    ): 
        self.id = None
        return 


    def __call__(
            self, 
            X: Tensor
    ) -> Tensor: 
        return self.forward(X)
    

    def __str__(
            self
    ): 
        return self.id
    

    def forward(
            self, 
            X: Tensor
    ) -> Tensor: 
        return





class GaussianNoise(Attack): 


    def __init__(
            self, 
            sigma: float
    ): 
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.id = "gaussian_noise"
        return
    
    
    def forward(
            self, 
            X: Tensor
    ) -> Tensor: 
        noise = normal(mean = 0., std = self.sigma, size = X.size(), requires_grad = False)
        return X + noise 
    




class GaussianBlur(Attack): 


    def __init__(
            self, 
            kernel_size: int
    ): 
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.id = "gaussian_blur"
        return
    

    def forward(
            self, 
            X: Tensor
    ) -> Tensor: 
        return gaussian_blur(X, self.kernel_size)





class Negative(Attack): 


    def __init__(
            self
    ): 
        super(Negative, self).__init__()
        self.id = "negative"
        return
    

    def forward(
            self, 
            X: Tensor
    ) -> Tensor: 
        return -X





class Black(Attack): 

    
    def __init__(
            self
    ): 
        super(Black, self).__init__()
        self.id = "black"
        return 
    

    def forward(
            self,
            X: Tensor
    ) -> Tensor: 
        return zeros_like(X, requires_grad = False)
    




class RandomCancellation(Attack): 
    
    
    def __init__(
            self,
            proba: float = 0.5
    ): 
        super(RandomCancellation, self).__init__()
        self.id = "random_cancellation"
        self.p = proba
        return 
    

    def forward(
            self, 
            X: Tensor
    ) -> Tensor: 
        rand_mask = rand_like(X) > self.p 
        X_new = X.clone()
        X_new[rand_mask] = 0.
        return X_new





class ShiftEmbedding(Attack): 
    
    
    def __init__(
            self,
            proba: float = 0.5
    ): 
        super(ShiftEmbedding, self).__init__()
        self.id = "shift_embedding"
        self.p = proba
        return 
    
    
    @no_grad()
    def forward(
            self, 
            X: Tensor
    ) -> Tensor: 
        shift = rand_like(X)
        shift = X[X != 0.].norm(keepdim = True) * shift / shift.norm(keepdim = True)
        mask = rand(X.size(1), device = X.device).type(X.dtype) <= self.p 
        shift[:, mask] = 0.
        return X + shift






