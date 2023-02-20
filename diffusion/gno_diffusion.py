import torch
from timeit import default_timer
from tqdm.auto import tqdm

from diffusion.loss import SpectralLoss, DiscreteLoss
from util.gaussian_process import GaussianProcess
from util.config import load_config
from util.tools import dataset_to_graphs


class GNODiffusionModel:
    """ An implementation of the function-space diffusion model using the Graph Neural Operator as the base architechture.
    See: https://arxiv.org/abs/2003.03485 and https://github.com/neuraloperator/graph-pde
    """
    def __init__(self, model, config_path):
        self.config = load_config(config_path)

        self.model = model

        self.device = self.config['device']
        self.dtype = self.config['dtype']
        self.support = self.config['support'].to(self.device)
        self.graph_radius = self.config['graph_radius']

        self.gp = GaussianProcess(self.config['kernel'], covar_eps=self.config['gp_covar_eps'])

        if self.config['loss'] == 'spectral':
            self.loss_fxn = SpectralLoss(self.config['kernel'], self.config['dist'], n_terms=self.config['n_terms'],
                                         support=self.support, dtype=self.dtype, device=self.device)
        elif self.config['loss'] == 'discrete':
            self.loss_fxn = DiscreteLoss(self.config['kernel'], self.config['support'], device=self.device)
        else:
            raise NotImplementedError('Loss not recognized')

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
        batch_size = u_0.shape[0]
        u_0 = u_0.to(self.device)

        if support is None:
            support = self.config['support']

        # Sample Gaussian noise on support
        xi = self.gp.sample(support, n_samples=batch_size)  # (batch_size, n_x)
        xi = xi.unsqueeze(-1).to(self.device)

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
        """
        Training loop for when our model is the GNO
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['optim_params']['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        if return_loss: train_losses = []
        for ep in range(1, self.config['epochs'] + 1):
            self.model.train()
            t1 = default_timer()
            train_l2 = 0
            for batch in train_loader:
                x, u_0 = batch
                assert len(u_0.shape) == 3, 'Input data is expected to have shape (batch_size, n_x, d)'
                assert x.shape == u_0.shape, 'Shape mismatch between x and u_0'
                batch_size = u_0.shape[0]
                u_0 = u_0.to(self.device).to(self.dtype)  # (batch_size, n_x, d)
                x = x.to(self.device).to(self.dtype)

                # Uniform on {1, 2, ..., max_t}
                t = torch.randint(1, self.config['max_t'] + 1, size=[batch_size], device=self.device)  # (batch_size, )
                u_t, xi = self.simulate_fwd_process(u_0, t, return_noise=True)

                # create a batch of graphs where y=u_t -- include t in node/edge features?
                graph_dataloader = dataset_to_graphs(x, u_t, data_t=t.unsqueeze(-1),
                                                     mode='batch', scale_t=self.config['max_t'], r=self.graph_radius)

                optimizer.zero_grad()
                out = self.model(graph_dataloader).reshape(batch_size, -1)  # (batch_size, n_x)
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
        u_t = self.gp.sample(query_points=support, n_samples=n_samples)
        u_t = u_t.unsqueeze(-1).to(self.device)  # (n_samples, n_x, 1)

        if return_path:
            diffusion_path = torch.empty(self.config['max_t'], n_samples, support.shape[0])  # (max_t, n_samples, n_x)

        for t in tqdm(range(self.config['max_t'], 0, -1)):
            xi = self.gp.sample(support, n_samples=n_samples)  # (n_samples, n_x)
            xi = xi.unsqueeze(-1).to(self.device)
            t_tensor = torch.ones(n_samples, device=self.device, dtype=self.dtype) * t

            graph_dataloader = dataset_to_graphs(support.repeat(n_samples, 1, 1), u_t, data_t=t_tensor.unsqueeze(-1),
                                                 mode='batch', scale_t=self.config['max_t'], r=self.graph_radius)

            out = self.model(graph_dataloader)
            out = out.reshape(n_samples, -1, 1)

            c1 = self.betas[t] / torch.sqrt(1. - self.gammas[t])
            c2 = torch.sqrt(1. - self.betas[t])
            c3 = torch.sqrt(self.beta_tildes[t])
            u_t = (u_t - c1 * out) / c2 + c3 * xi

            if return_path:
                diffusion_path[t] = u_t

        if return_path:
            return u_t, diffusion_path
        else:
            return u_t

    @torch.no_grad()  # Disable gradient computations while sampling
    def sample_conditional(self, x_cond, y_cond, n_samples=1, return_path=False, query_points=None, n_free=0):
        """ Samples from the GNO model conditioned on some function observations.
        Note that x_cond and y_cond do not need to be in query_points or in the support of the model training set.

         x_cond: (m, d_x)
         y_cond: (m, d_y)
         n_samples: how many samples to draw, starting from the same conditioning information
         query_poitns: where to query the resulting curves; assumed to be the same for all samples
         n_free: how many diffusion steps to run without conditioning; these are the last n_free steps
        """

        if query_points is None:
            query_points = self.support
        n_cond = x_cond.shape[0]
        # Removing duplicates from query points and x_cond
        repeat_mask = torch.ones(query_points.shape[0], dtype=torch.bool)
        for i, x in enumerate(query_points):
            if torch.any(torch.isclose(x, x_cond)):
                repeat_mask[i] = False
        # All generation is done on augmented_support, which contains both x_cond and query_points (without dupes)
        augmented_support = torch.vstack([query_points[repeat_mask], x_cond])

        y_cond = y_cond.unsqueeze(0)

        # Initial sample u_T ~ GP(0, k)
        u_t = self.gp.sample(query_points=augmented_support, n_samples=n_samples)
        u_t = u_t.unsqueeze(-1).to(self.device)  # (n_samples, n_x, 1)

        if return_path:
            diffusion_path = torch.empty(self.config['max_t'], n_samples, augmented_support.shape[0])  # (max_t, n_samples, n_x)

        for t in tqdm(range(self.config['max_t'], 0, -1)):
            # Unconditionally generate u_t on augmented support
            xi = self.gp.sample(augmented_support, n_samples=n_samples)  # (n_samples, n_x)
            xi = xi.unsqueeze(-1).to(self.device)
            t_tensor = torch.ones(n_samples, device=self.device, dtype=self.dtype) * t

            graph_dataloader = dataset_to_graphs(augmented_support.repeat(n_samples, 1, 1), u_t,
                                                 data_t=t_tensor.unsqueeze(-1), mode='batch',
                                                 scale_t=self.config['max_t'], r=self.graph_radius)
            out = self.model(graph_dataloader)
            out = out.reshape(n_samples, -1, 1)

            c1 = self.betas[t] / torch.sqrt(1. - self.gammas[t])
            c2 = torch.sqrt(1. - self.betas[t])
            c3 = torch.sqrt(self.beta_tildes[t])
            u_t = (u_t - c1 * out) / c2 + c3 * xi  # (n_samples, n_x, d_x)

            # Apply conditioning information, if desired
            if t > n_free:
                # Diffuse conditioning information via forward process
                y_cond_t = self.simulate_fwd_process(y_cond, t, return_noise=False, support=x_cond)
                # Set generated curve to y_cond_t at x_cond (subset of augmented_support)
                u_t[:, -n_cond:] = y_cond_t

            if return_path:
                diffusion_path[t] = u_t

        # Sort augmented_support, and sort u_t according to this permutation
        # NOTE: the logic below will not work when d_x > 1
        augmented_support = augmented_support.flatten()
        augmented_support_sorted, sort_idxs = torch.sort(augmented_support)
        u_t = u_t[:, sort_idxs, :]

        if return_path:
            return u_t, augmented_support_sorted, diffusion_path
        else:
            return u_t, augmented_support_sorted
