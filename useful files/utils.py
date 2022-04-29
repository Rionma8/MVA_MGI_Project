
def get_data_inverse_scaler(config):
    if config.center_image and config.is_image:
        return lambda x: (x + 1.) / 2.  # Rescale from [-1, 1] to [0, 1]
    else:
        return lambda x: x

def build_beta_fn(config):
    def beta_fn(t):
        return config.beta0 + config.beta1 * t
    return beta_fn


def build_beta_int_fn(config):
    def beta_int_fn(t):
        return config.beta0 * t + 0.5 * config.beta1 * t**2
    return beta_int_fn

def add_dimensions(x, is_image):
    if is_image:
        return x[:, None, None, None]
    else:
        return x[:, None]