from numpy import ndarray, arange, mean, log, sqrt, std, array, zeros
from numpy.random import choice, uniform 

from torch import Tensor, tensor, diag, tril, eye, stack, cat, zeros_like
from torch.linalg import cholesky, solve
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sklearn.cluster import KMeans

from src.data.base import BaseDataset
from src.federated_learning.base import BaseClient, BaseServer

from typing import Union, Any

from tqdm import tqdm

from logging import getLogger, basicConfig, INFO, StreamHandler

from warnings import simplefilter 
simplefilter("ignore")




basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[StreamHandler()]  
)

logger = getLogger("FLDETECTOR/Server")





class Client(BaseClient): 


    def __init__(
            self,
            train_dataset: BaseDataset,
            test_dataset: BaseDataset,
            distribution: ndarray,
            batch_size: int,
            device: str
    ): 
        super(Client, self).__init__(
            train_dataset, test_dataset,
            distribution, batch_size,
            device
        )
        return
    

    def train(
            self,
            num_epochs: int,
            model: Module, 
            criterion: Module,
            optimizer_class: Optimizer,
            optimizer_params: dict[str, Any]
    ) -> dict[str, Any]: 
        model, criterion, optimizer = self.training_setup(model, criterion, optimizer_class, optimizer_params)
        cumulative_loss = 0.

        for _ in range(num_epochs): 
            for X, y in self._train_dataloader: 
                y_hat = model(X.to(self.device))
                loss = criterion(y_hat, y.to(self.device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                cumulative_loss += loss.item()

        self.clear_cache()

        training_dict = {
            "model": [p.data.cpu() for p in model.parameters()],
            "loss": cumulative_loss / self._num_samples,
            "num_samples": self._num_samples
        }

        return training_dict





class Server(BaseServer): 


    def __init__(
            self,
            clients: list[BaseClient],
            participation_rate: Union[int, float],
            model: Module,
            window_size: int = 10,
            batch_size: int = 20,
            iter_max: int = 10
    ): 
        super(Server, self).__init__(
            clients = clients,
            participation_rate = participation_rate
        )
        self.model = model 
        self.W = [cat([p.data.view(-1) for p in self.model.parameters()])]
        self.dW = []
        self.ddW = []
        self.N = window_size
        self.B = batch_size
        self.iter_max = iter_max
        self.benign_clients = [client for client in self.clients]
        self.malicious_clients = []
        self.past_scores = []
        if isinstance(self.participation_rate, int): 
            self.K = self.participation_rate
        else:
            self.K = int(len(self.clients) * self.participation_rate)
        return 
    

    def _hessian_estimation(
            self
    ) -> Tensor:
        # Setup
        if len(self.ddW) < self.N:
            return None
        
        # Vector v = w_t - w_{t-1}, shape: (1, p)
        v = self.dW[-1].view(1, -1)  # shape: (1, p)

        # Stack the N most recent differences
        dW = stack(self.dW, dim = 1)  # shape: (p, N)
        dG = stack(self.ddW, dim = 1)  # shape: (p, N)

        # Compute required matrices
        dWdW = dW.T @ dW            # shape: (N, N)
        dWdG = dW.T @ dG            # shape: (N, N)
        D = diag(diag(dWdG))    # D: diagonal of dWdG
        L = tril(dWdG, diagonal=-1) # L: strictly lower triangular of dWdG

        # Compute scalar sigma
        s_last = self.dW[-1].view(-1)  # shape: (p,)
        y_last = self.ddW[-1].view(-1) # shape: (p,)
        sigma = (y_last @ s_last) / (s_last @ s_last + 1e-12)

        # Compute J using Cholesky
        A = sigma * dWdW + L @ D @ L.T  # shape: (N, N)
        J = cholesky(A + 1e-8 * eye(A.size(0), device=A.device))  # for numerical stability

        # Construct RHS vector
        rhs = cat([
            dG.T @ v.T,                    # shape: (N, 1)
            sigma * (dW.T @ v.T)          # shape: (N, 1)
        ], dim=0)                         # shape: (2N, 1)

        # Construct inverse transformation matrices
        upper = cat([-(D + 1e-8).sqrt(), (D + 1e-8).rsqrt() @ L.T], dim=1)  # (N, 2N)
        lower = cat([zeros_like(J), J.T], dim=1)    # (N, 2N)
        Q1 = cat([upper, lower], dim=0)                   # (2N, 2N)

        lower = cat([(D + 1e-8).sqrt(), zeros_like(J)], dim=1)
        upper = cat([(D + 1e-8).rsqrt() @ L.T, J], dim=1)
        Q2 = cat([lower, upper], dim=0)

        # Solve the system
        q = solve(Q1, rhs)
        q = solve(Q2, q)  # final q

        # Final Hessian-vector product
        Hv = sigma * v - (dG @ q[:self.N] + sigma * dW @ q[self.N:]).T.view(1, -1)
        return Hv.view(-1)
    

    def _suspicious_scores(
        self,
        predicted: list[Tensor],
        received: list[Tensor]
    ) -> list[Tensor]:
        distances = [(g_hat - g).norm() for g_hat, g in zip(predicted, received)]
        d_t = stack(distances)
        d_t_norm = d_t / d_t.sum()  # normalize over all clients

        # Append and maintain window
        self.past_scores.append(d_t_norm)
        if len(self.past_scores) > self.N:
            self.past_scores.pop(0)

        # Compute final suspicious score for each client
        s = stack(self.past_scores).mean(dim=0)
        return s.tolist()  # suspicious score per client
    

    def _gap_statistic(
            self, 
            scores: list[float]
        ) -> int:
        data = array(scores).reshape(-1, 1)
        ref_disps = zeros((self.B, self.iter_max))

        for b in range(self.B):
            ref = uniform(0, 1, size=data.shape)
            for k in range(1, min(self.iter_max + 1, data.shape[0])):
                km = KMeans(n_clusters=k, n_init="auto").fit(ref)
                ref_disps[b, k - 1] = km.inertia_

        gaps = []
        for k in range(1, min(self.iter_max + 1, data.shape[0])):
            km = KMeans(n_clusters=k, n_init="auto").fit(data)
            orig_disp = km.inertia_ + 1e-8
            ref_disp_log = log(ref_disps[:, k - 1] + 1e-8)
            gap = mean(ref_disp_log) - log(orig_disp)
            sk = sqrt(1 + 1/self.B) * std(ref_disp_log)
            gaps.append((k, gap, sk))

        for i in range(len(gaps) - 1):
            if gaps[i][1] >= gaps[i + 1][1] - gaps[i + 1][2]:
                return gaps[i][0]

        return 1 
    

    def aggregation(
            self, 
            updates: list[Any]
        ):
        # Setup 
        num_samples = sum([update["num_samples"] for update in updates])

        for p in self.model.parameters():
            p.data.zero_()

        for update in updates:
            w = update["num_samples"] / num_samples 
            for p_new, p in zip(update["model"], self.model.parameters()):
                p.data.add_(w * p_new)
        
        return
    

    def train(
            self, 
            num_rounds: int,
            num_local_epochs: int,
            criterion: Module, 
            optimizer_class: Optimizer, 
            optimizer_params: dict[str, Any],
            evaluation_step: int,
            metrics: dict[str, Module]
    ) -> dict[str, Any]: 
        # Setup
        training_dict = dict()
        evaluation_dict = dict()

        # Training
        for r in tqdm(range(1, num_rounds + 1)): 
            # Round
            updates = []
            received = []
            
            indexes = choice(arange(len(self.benign_clients)), size = self.K, replace = False)
            clients = [self.benign_clients[i] for i in indexes] 
            for client in clients: 
                update = client.train(num_local_epochs, self.model, criterion, optimizer_class, optimizer_params)
                updates.append(update)
                received.append(cat([p.view(-1) for p in update["model"]]))

            # Detector 
            W_t = cat([p.data.view(-1) for p in self.model.parameters()])
            new_dW = W_t - self.W[-1]
            new_ddW = new_dW - self.dW[-1] if len(self.dW) > 0 else new_dW

            self.W.append(W_t)
            self.dW.append(new_dW)
            self.ddW.append(new_ddW)

            if len(self.W) > self.N: 
                self.W.pop(0)
            if len(self.dW) > self.N:
                self.dW.pop(0)
            if len(self.ddW) > self.N:
                self.ddW.pop(0)

            selected_updates = updates

            if len(self.dW) == self.N and len(self.ddW) == self.N:
                predicted = [W_t - self._hessian_estimation() for _ in received]
                scores = self._suspicious_scores(predicted, received)
                k = self._gap_statistic(scores)

                if k > 1:
                    km = KMeans(n_clusters=k, n_init="auto").fit(array(scores).reshape(-1, 1))
                    majority_cluster = max(set(km.labels_), key=list(km.labels_).count)
                    selected_updates = []
                    for idx, label in zip(indexes, km.labels_):
                        if label == majority_cluster:
                            selected_updates.append(updates[indexes.tolist().index(idx)])
                        else:
                            self.malicious_clients.append(idx)
                            logger.warning(f"Client {idx} flagged as malicious and removed.")
                    self.benign_clients = [client for i, client in enumerate(self.clients) if i not in self.malicious_clients]
                    self.W = [cat([p.data.view(-1) for p in self.model.parameters()])]
                    self.dW = []
                    self.ddW = []

            # Aggregation
            self.aggregation(selected_updates)

            # Tracking
            num_samples = sum([update["num_samples"] for update in updates])
            training_loss = sum([update["loss"] * update["num_samples"] / num_samples for update in updates])
            training_dict[r] = {"loss": training_loss}

            logger_msg = f"Round {r}: training_loss = {round(training_loss, 4)}"

            if r % evaluation_step == 0: 
                evaluation_dict[r] = self.evaluate(criterion, metrics)

                logger_msg += f", evaluation_loss = {round(evaluation_dict[r]['server']['loss'], 4)}"
                for metric_name, metric_val in evaluation_dict[r]["server"]['metrics'].items():
                    logger_msg += f", {metric_name} = {round(metric_val, 4)}"

                logger.info(logger_msg)

            else: 
                logger.info(logger_msg)

        # Results
        tracking_dict = {
            "training": training_dict,
            "evaluation": evaluation_dict
        }
            
        return tracking_dict
    

    def evaluate(
            self, 
            criterion: Module,
            metrics: dict[str, Module]
    ) -> dict[str, Any]: 
        # Setup 
        num_samples = 0.
        evaluation_dict = {
            "server": {
                "loss": 0.,
                "metrics": {
                    metric_name: 0 for metric_name in metrics.keys()
                }
            },
            "clients": dict()
        }

        # Evaluation
        for i, client in enumerate(self.clients): 
            client_evaluation_dict = client.evaluate(self.model, criterion, metrics)
            client_num_samples = client.num_samples()

            evaluation_dict["clients"][i] = client_evaluation_dict

            evaluation_dict["server"]["loss"] += client_num_samples * client_evaluation_dict["loss"]
            for metric_name, metric_value in client_evaluation_dict["metrics"].items():
                evaluation_dict["server"]["metrics"][metric_name] += client_num_samples * metric_value

            num_samples += client_num_samples

        evaluation_dict["server"]["loss"] /= num_samples 
        for metric_name, metric_val in evaluation_dict["server"]["metrics"].items():
            evaluation_dict["server"]["metrics"][metric_name] = metric_val / num_samples

        return evaluation_dict



