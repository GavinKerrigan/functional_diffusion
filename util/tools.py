import torch
import torch.distributions as d
import scipy
import yaml
from tqdm.auto import tqdm
import numpy as np
# from sklearn.metrics import pairwise_distances
import torch_geometric.data


def gaussian_kl(mu1, cov1, mu2, cov2):
    """
    Computes the KL divergence between two finite dim'l Gaussian distributions.

    mu_1, mu_2: (d,)
    cov_1, cov_2: (d,d)
    """

    assert len(mu1.shape) == len(mu2.shape) == 1, f'Means should be 1d; got shapes {mu1.shape} {mu2.shape}'
    assert len(cov1.shape) == len(cov2.shape) == 2, f'Covs should be 2d; got shapes {cov1.shape} {cov2.shape}'

    cov1 = cov1.double()
    cov2 = cov2.double()

    gaussian1 = d.MultivariateNormal(mu1, cov1)
    gaussian2 = d.MultivariateNormal(mu2, cov2)

    return d.kl.kl_divergence(gaussian1, gaussian2)


def gaussian_2_wasserstein(mu1, cov1, mu2, cov2):
    """
    Computes the 2-Wasserstein distance between two finite dim'l Gaussian distributions.

    mu_1, mu_2: (d,)
    cov_1, cov_2: (d,d)
    """
    assert len(mu1.shape) == len(mu2.shape) == 1, f'Means should be 1d; got shapes {mu1.shape} {mu2.shape}'
    assert len(cov1.shape) == len(cov2.shape) == 2, f'Covs should be 2d; got shapes {cov1.shape} {cov2.shape}'

    term1 = torch.linalg.norm(mu1 - mu2) ** 2

    sqrt_cov2 = scipy.linalg.sqrtm(cov2)
    sqrt_cov2 = torch.as_tensor(sqrt_cov2, dtype=torch.double)
    term2 = sqrt_cov2 @ cov1 @ sqrt_cov2
    term2 = scipy.linalg.sqrtm(term2)

    term3 = torch.trace(cov1 + cov2 - 2 * term2)

    w2 = torch.sqrt(term1 + term3)
    return w2


def fwd_process_diagnostic(diffusion_model, u_0):
    """
    Simulate forward process for values values of t, and measure how far away from the prior we are.
    Useful for checking that the forward process is converging (close) to the prior.
    Note the simulation and metrics are evaluated on a uniform grid.

    Metrics:
    (1) KL divergence: fwd, reverse, symmetrized.
    (3) 2-Wasserstein distance.
    """
    batch_size = u_0.shape[0]
    n_x = u_0.shape[1]
    x_grid = torch.linspace(0, 1, n_x, dtype=torch.double).reshape(-1, 1)

    gp_mean = torch.zeros(n_x, dtype=torch.double).cpu()
    gp_cov = diffusion_model.config['kernel'](x_grid, x_grid).double().cpu()  # Covariance matrix on grid

    max_t = diffusion_model.config['max_t']
    fwd_kl_vals = torch.empty(max_t)
    rev_kl_vals = torch.empty(max_t)
    w2_vals = torch.empty(max_t)
    mean_err_vals = torch.empty(max_t)
    cov_matr_err_vals = torch.empty(max_t)

    for t in tqdm(range(max_t)):
        # Simulate forward process
        # NOTE: this is only sampling one u_t value per datapoint. May want to do more for stability.
        t_tensor = (torch.ones(batch_size) * t).to(torch.long).to(diffusion_model.device)
        u_t = diffusion_model.simulate_fwd_process(u_0, t_tensor)

        u_t_mean = torch.mean(u_t.squeeze(), axis=0).cpu()
        u_t_cov = torch.cov(u_t.squeeze().T).cpu() + diffusion_model.config['gp_covar_eps'] * \
                  torch.eye(n_x, device='cpu')  # NOTE: adding eps here for stability

        fwd_kl_vals[t] = gaussian_kl(gp_mean, gp_cov, u_t_mean, u_t_cov)
        rev_kl_vals[t] = gaussian_kl(u_t_mean, u_t_cov, gp_mean, gp_cov)
        w2_vals[t] = gaussian_2_wasserstein(gp_mean.double(), gp_cov.double(), u_t_mean.double(), u_t_cov.double())
        mean_err_vals[t] = torch.linalg.norm(u_t_mean - gp_mean)
        cov_matr_err_vals[t] = torch.linalg.norm(u_t_cov - gp_cov)

    return fwd_kl_vals, rev_kl_vals, w2_vals, mean_err_vals, cov_matr_err_vals


def batched_autocorrelation(x, normalize=False):
    """
    Computes autocorrelation for a batch of curves
    x has shape (batch_size, seq_length)
    """
    x = np.asarray(x)
    assert len(x.shape) == 2, f'Input has shape {x.shape} -- expected (batch_size, n_x)'

    seq_length = x.shape[1]

    if normalize:
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)
        x = (x - x_mean) / (np.sqrt(seq_length) * x_std)

    autocorr = np.empty_like(x)
    for i, row in enumerate(x):
        autocorr[i] = np.correlate(row, row, mode='full')[seq_length - 1:]

    return autocorr


class MinMaxNormalizer:
    def __init__(self, scale=3.):
        self.scale = scale
        self.data_min = None
        self.data_max = None

    def set_params(self, min_val, max_val):
        # Sets the parameters of the normalizer from args
        self.data_min = min_val
        self.data_max = max_val

    def fit(self, data):
        # data: (batch_size, n_x, d)
        self.data_min = torch.min(data)
        self.data_max = torch.max(data)

    def normalize(self, data):
        # data: (batch_size, n_x, d)
        rescaled_data = (data - self.data_min) / (self.data_max - self.data_min)
        rescaled_data = 2 * self.scale * rescaled_data - self.scale
        return rescaled_data

    def unnormalize(self, data):
        # data: (batch_size, n_x, d)
        assert self.data_min is not None, 'Must call normalize before unnormalizing'
        assert self.data_max is not None, 'Must call normalize before unnormalizing'

        rescaled_data = (data + self.scale) * (self.data_max - self.data_min) / (2 * self.scale) + self.data_min
        return rescaled_data


def save_everything(model, config_path, out_path, scaler=None, model_arch=None):
    # model is the actual pytorch model
    # fpath is the filepath (plus extension) of where we are saving
    torch.save(model.state_dict(), out_path)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if scaler is not None:
        config['scaler_params'] = [scaler.data_min, scaler.data_max]

    if model_arch is not None:
        config['model_arch'] = model_arch

    out_path_config = out_path[-2:] + '_config.yml'
    with open(out_path_config, 'w') as file:
        yaml.dump(config, out_path_config)


def curve_to_graph(x, y, t=None, r=0.1):
    """
    NOTE: This is not batched -- only handles one curve at a time

    x: (n_x, d_x)
    y: (n_x, d_y)
    r: scalar representing radius of connectivity; l2 distance
    """
    assert x.shape[0] == y.shape[0]
    assert x.dim() == y.dim() == 2

    n_x = x.shape[0]
    d_x = x.shape[1]
    d_y = y.shape[1]
    device = x.device

    # Get graph connectivity in COO Format
    # pwd = pairwise_distances(x)  # (n_x, n_x)
    pwd = torch.cdist(x, x)  # (n_x, n_x)
    # edge_index = np.vstack(np.where(pwd <= r))  # (2, n_edges)
    edge_index = torch.vstack(torch.where(pwd <= r))
    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    n_edges = edge_index.shape[1]

    # Create edge attributes: e(x1, x2) = (x1, x2, y[x1], y[x2])
    # or if t is available: e(x1, x2) = (x1, x2, y[x1], y[x2], t)
    n_edge_features = 2 * d_x + 2 * d_y
    if t is not None:
        n_edge_features += 1
    edge_attr = torch.empty(n_edges, n_edge_features, device=device)
    edge_attr[:, :2 * d_x] = x[edge_index].transpose(1, 0).reshape(n_edges, -1)
    edge_attr[:, 2 * d_x:2 * d_x + 2 * d_y] = y[edge_index].transpose(1, 0).reshape(n_edges, -1)
    if t is not None:
        edge_attr[:, -1] = t

    # Create node features: v(x) = (x, y[x])
    # or if t is available: v(x) = (x, y[x], t)
    n_node_features = d_x + d_y
    if t is not None:
        n_node_features += 1
    node_features = torch.empty(n_x, n_node_features, device=device)
    node_features[:, :d_x] = x
    node_features[:, d_x:d_x + d_y] = y
    if t is not None:
        node_features[:, -1] = t

    # note: can also pass y=y in here but it isn't used; can cut down on memory by not doing this
    graph = torch_geometric.data.Data(x=node_features,
                                      edge_index=edge_index,
                                      edge_attr=edge_attr)

    return graph


def dataset_to_graphs(data_x, data_y, data_t=None, batch_size=128, shuffle=False, r=0.1, mode='batch', scale_t=1.):
    """ Converts a dataset of curves into a dataset of graphs

    data_x: (n_data, n_x, d_x)
    data_y: (n_data, n_x, d_y)
    data_t: (n_data, 1)  -- optional time data
    r: connectiivity radius
    mode: 'batch' or 'dataloader' -- whether to batch the graphs or make them into a dataloader
    scale_t: factor by which we scale data_t -- typically is max_t
    """
    assert data_x.shape[0] == data_y.shape[0], 'Mismatch in num. datapoints'
    assert data_x.shape[1] == data_y.shape[1], 'Mismatch between n_x in x and y'
    assert data_x.dim() == data_y.dim() == 3, 'Data should be 3-diml'
    if data_t is not None:
        assert data_t.shape[0] == data_x.shape[0], 'Mismatch between t and x'
        assert data_t.shape[1] == 1, 'data_t should have shape (n_x, 1)'

    data_graph = []
    for i in range(data_x.shape[0]):
        if data_t is not None:
            t = data_t[i] / scale_t
        else:
            t = None
        graph = curve_to_graph(data_x[i], data_y[i], t=t, r=r)
        data_graph.append(graph)

    if mode == 'dataloader':
        out = torch_geometric.loader.DataLoader(data_graph, batch_size=batch_size, shuffle=shuffle)
    elif mode == 'batch':
        out = torch_geometric.data.Batch().from_data_list(data_graph)

    return out


def make_ddx_operator(n, dx=None):
    # Creates a matrix approximation to the operator d/dx (in one dimension) via finite differences
    # n: number of discretization points
    # dx: array where dx[j] = x[j+1] - x[j]; length n-1
    if dx is None:
        dx = torch.ones(n - 1, dtype=float) / n

    operator = torch.zeros((n, n))

    # Forward difference at LHS boundary
    operator[0, 0] = -1. / dx[0]
    operator[0, 1] = 1. / dx[0]

    # Central differences on interior
    for j in range(1, n - 1):
        operator[j, j - 1] = -1. / (dx[j] + dx[j - 1])
        operator[j, j + 1] = 1. / (dx[j] + dx[j - 1])

    # Backward difference at RHS boundary
    operator[n - 1, -2] = -1. / dx[-1]
    operator[n - 1, -1] = 1. / dx[-1]

    return operator
