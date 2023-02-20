import torch
from timeit import default_timer
from tqdm.auto import tqdm
from util.config import load_config


class DDPM:
    """
    An implemenation of the vanilla DDPM model.
    """

    def __init__(self, model, config_path):
        self.config = load_config(config_path)

        self.model = model

        self.device = self.config['device']
        self.dtype = self.config['dtype']
        self.support = self.config['support']

        gaussian_mean = torch.zeros(self.config['n_x'], device=self.device)
        gaussian_cov = torch.eye(self.config['n_x'], device=self.device)
        self.gaussian = torch.distributions.MultivariateNormal(gaussian_mean, gaussian_cov)

        self.loss_fxn = lambda _model, _xi: torch.linalg.norm(_model - _xi, axis=1) ** 2

        self.betas, self.beta_tildes, self.gammas = self.construct_diffusion_params(beta_min=self.config['beta_min'],
                                                                                    beta_max=self.config['beta_max'],
                                                                                    max_t=self.config['max_t'],
                                                                                    device=self.device)
        return

    def construct_diffusion_params(self, beta_min=1e-4, beta_max=0.02, max_t=1000, device='cpu'):
        """
        Creates various parameters (beta, beta_tilde, gamma) used in the diffusion process.
        """
        # Linearly interpolate betas between beta_min and beta_max
        betas = torch.zeros(max_t + 1, requires_grad=False).to(device)
        betas[1:] = torch.linspace(beta_min, beta_max, max_t)

        beta_tildes = torch.zeros(max_t + 1, requires_grad=False).to(device)
        gammas = torch.ones(max_t + 1, requires_grad=False).to(device)
        for t in range(1, max_t + 1):
            gammas[t] = (1. - betas[t]) * gammas[t - 1]
            beta_tildes[t] = (1. - gammas[t - 1]) / (1. - gammas[t]) * betas[t]
        return betas, beta_tildes, gammas

    def simulate_fwd_process(self, u_0, t, return_noise=False, support=None):
        """
        Simulates the forward process for t steps with initial values u_0.
        u_0 : (batch_size, n_x, d) starting points for diffusion process
        t: (batch_size,) array of diffusion times
        """
        assert len(u_0.shape) == 3, 'Input data is expected to have shape (batch_size, n_x, d)'
        assert u_0.shape[1] == self.config[
            'n_x'], f'Input data has shape {u_0.shape} -- expected dim1={self.config["n_x"]}'
        assert u_0.shape[2] == 1, f'Input data has shape {u_0.shape} -- d=1 for scalar DDPM'
        batch_size = u_0.shape[0]
        u_0 = u_0.to(self.device)

        if support is None:
            support = self.config['support']

        # Sample Gaussian noise on support
        xi = self.gaussian.sample([batch_size])  # (batch_size, n_x)
        xi = xi.unsqueeze(-1)

        # Construct u_t, perturbed function values on grid
        scaled_init_fxn = torch.sqrt(self.gammas[t]).reshape(-1, 1, 1) * u_0
        scaled_noise = torch.sqrt(1. - self.gammas[t]).reshape(-1, 1, 1) * xi
        u_t = scaled_init_fxn + scaled_noise  # (batch_size, n_x, d)

        assert u_0.shape == u_t.shape, 'u_t should have same shape as u_0'
        if return_noise:
            return u_t, xi
        else:
            return u_t

    def train(self, train_loader, test_loader=None, logger=None, return_loss=False):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['optim_params']['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        if return_loss: train_losses = []
        for ep in range(1, self.config['epochs'] + 1):
            self.model.train()
            t1 = default_timer()
            train_l2 = 0
            for u_0, in train_loader:
                assert len(u_0.shape) == 3, 'Input data is expected to have shape (batch_size, n_x, d)'
                batch_size = u_0.shape[0]
                u_0 = u_0.to(self.device).to(self.dtype)  # (batch_size, n_x, d)

                # Uniform on {1, 2, ..., max_t}
                t = torch.randint(1, self.config['max_t'] + 1, size=[batch_size], device=self.device)  # (batch_size, )
                u_t, xi = self.simulate_fwd_process(u_0, t, return_noise=True)

                optimizer.zero_grad()
                out = self.model(u_t, t).squeeze(-1)  # (batch_size, n_x)
                # out = self.model(u_t, t, self.gammas[t]).squeeze(-1)  # (batch_size, n_x)

                loss = torch.mean(self.loss_fxn(xi.squeeze(-1), out))
                loss.backward()

                optimizer.step()
                train_l2 += loss.item()
            scheduler.step()

            # Evaluation loop if we have an eval set
            test_l2 = torch.nan
            if test_loader is not None:
                with torch.no_grad():
                    self.model.eval()
                    test_l2 = 0.0
                    for u_0, in train_loader:
                        # forward pass
                        batch_size = u_0.shape[0]
                        u_0 = u_0.to(self.device).to(self.dtype)

                        # Uniform on {1, 2, ..., max_t}
                        t = torch.randint(1, self.config['max_t'] + 1, size=[batch_size], device=self.device)
                        u_t, xi = self.simulate_fwd_process(u_0, t, return_noise=True)

                        out = self.model(u_t, t).squeeze(-1)
                        loss = torch.mean(self.beta_tildes[t] * self.loss_fxn(xi.squeeze(-1), out))
                        test_l2 += loss.item()

            train_l2 /= len(train_loader)
            if return_loss: train_losses.append(train_l2)
            if test_loader is not None: test_l2 /= len(test_loader)
            t2 = default_timer()

            if logger:
                logger.debug(
                    f'Epoch {ep}/{self.config["epochs"]} | Train {train_l2:.4f} | Test {test_l2:.4f} | Sec. per epoch {t2 - t1:.2f} |')

            # if ep % 10 == 0 or ep == 1:
            if ep % 1 == 0:
                print(
                    f'Epoch {ep}/{self.config["epochs"]} | Train {train_l2:.4f} | Test {test_l2:.4f} | Sec. per epoch {t2 - t1:.2f} |')

        if return_loss:
            return train_losses

    @torch.no_grad()  # Disable gradient computations while sampling
    def sample(self, n_samples=1, return_path=False, support=None):
        """
        Samples the reverse diffusion process.
        """
        if support is None:
            support = self.support

        # Initial sample u_T ~ GP(0, k)
        u_t = self.gaussian.sample([n_samples])
        # u_t = u_t.unsqueeze(-1).to(self.device)  # (n_samples, n_x, 1)

        if return_path:
            diffusion_path = torch.empty(self.config['max_t'], n_samples, 2)  # (max_t, n_samples, n_x)

        for t in tqdm(range(self.config['max_t'], 0, -1)):
            if t > 1:
                xi = self.gaussian.sample([n_samples])  # (n_samples, n_x)
            else:
                xi = torch.zeros((n_samples, 2), device=self.device)

            # xi = self.gaussian.sample([n_samples])  # (n_samples, n_x)

            # xi = xi.unsqueeze(-1).to(self.device)
            t_tensor = torch.ones(n_samples, device=self.device, dtype=self.dtype) * t

            # out = self.model(u_t, t_tensor).unsqueeze(-1)
            out = self.model(u_t, t_tensor)
            # print(f'mean out norm {torch.mean(torch.linalg.norm(out, axis=1))}')
            # print(f'out {out}')
            # out = self.model(u_t, t_tensor, self.gammas[t])

            c1 = 1. / torch.sqrt(1. - self.betas[t])
            c2 = self.betas[t] / torch.sqrt(1. - self.gammas[t])
            c3 = torch.sqrt(self.beta_tildes[t])
            u_t = c1 * (u_t - c2 * out) + c3 * xi

            if return_path:
                diffusion_path[t - 1] = u_t.squeeze()

        if return_path:
            return u_t, diffusion_path
        else:
            return u_t
