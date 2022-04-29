import torch

def get_model_fn(model, train=False):
    def model_fn(x, labels):
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn

def get_score_fn(config, sde, model, train=False):
    model_fn = get_model_fn(model, train=train)

    def score_fn(u, t):
        score = model_fn(u.type(torch.float64), t.type(torch.float64))
        noise_multiplier = sde.noise_multiplier(t).type(torch.float32)

        if config.mixed_score:
            if sde.is_augmented:
                _, z = torch.chunk(u, 2, dim=1)
                ones = torch.ones_like(z, device=config.device)
                var_zz = (sde.var(t, 0. * ones, (sde.gamma / sde.m_inv) * ones)[2]).type(torch.float32)
                return - z / var_zz + score * noise_multiplier
            else:
                ones = torch.ones_like(u, device=config.device)
                var = (sde.var(t, ones)[0]).type(torch.float32)
                return -u / var + score * noise_multiplier
        else:
            return noise_multiplier * score
    return score_fn