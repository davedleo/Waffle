from torch import Tensor, zeros_like, normal
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




