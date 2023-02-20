import torch
from util import kernel


def load_config(cfg_path):
    """ Parses a configuration .yml file into a dictionary.
    """
    import yaml
    with open(cfg_path, 'r') as file:
        config = yaml.safe_load(file)

    # Map device string to torch.device object
    if 'device' in config:
        config['device'] = torch.device(config['device'])
    else:  # Default to cpu
        config['device'] = torch.device('cpu')

    # Map dtype string to torch.dtype object
    if 'dtype' in config:
        if config['dtype'] == 'float':
            config['dtype'] = torch.float
        elif config['dtype'] == 'double':
            config['dtype'] = torch.double
        else:
            raise NotImplementedError
    else:  # Default to float
        config['dtype'] = torch.float

    # Construct the support -- assumed to be a uniform grid on [x_min, x_max]
    support = torch.linspace(config['x_min'], config['x_max'], config['n_x']).reshape(-1, 1)
    config['support'] = support.to(config['dtype'])

    # Make a distribution object -- assumed to be Unif[x_min, x_max]
    dist = kernel.UniformDistribution(config['x_min'], config['x_max'])
    config['dist'] = dist

    # Map kernel params to Kernel object
    if 'kernel' in config:
        if config['kernel']['name'] == 'exponential':
            k = kernel.ExponentialKernel(0, 1, config['kernel']['length_scale'], config['kernel']['var'])
        elif config['kernel']['name'] == 'matern32':
            k = kernel.Matern32(0, 1, config['kernel']['length_scale'], config['kernel']['var'])
        elif config['kernel']['name'] == 'rbf':
            k = kernel.RBF(0, 1, config['kernel']['length_scale'], config['kernel']['var'])
        else:
            raise NotImplementedError
    else:  # Default to an Exp(0.2, 1) kernel
        k = kernel.ExponentialKernel(0, 1, 0.2, 1)
    config['kernel'] = k

    return config
