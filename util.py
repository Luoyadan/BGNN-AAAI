import torch



def sample_normal(mu, log_variance, num_samples):
    '''
    :param mu: mean parameter of distribution (batch * 1 * node * node)
    :param log_variance:  log variance of distribution
    :param num_samples: number of samples to generate
    :return: tensor: samples from distribution of size (batch * num_samples * node * node)
    '''
    eps = torch.randn(mu.size(0), num_samples, mu.size(2), mu.size(3)).cuda()
    return mu + eps * torch.sqrt(torch.exp(log_variance))
