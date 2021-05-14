def create_model(instance_name, **kwargs):
    if instance_name == 'ContextualUNet':
        from .contextual_unet import ContextualUNet
        return ContextualUNet()
    elif instance_name == 'ContextualUNetV1':
        from .contextual_unet_v1 import ContextualUNetV1
        return ContextualUNetV1()
    elif instance_name == 'DCGan':
        from .dcgan import DCGan
        return DCGan()
    elif instance_name == 'ContextualUNetV2':
        from .contextual_unet_v2 import ContextualUNetV2
        return ContextualUNetV2()
    elif instance_name == 'NisslNet':
        from .nissl_net import NisslNet
        return NisslNet()
    elif instance_name == 'EncoderDecoder':
        from .encoder_decoder import EncoderDecoder
        n_pools, n_start_filters, kernel_sizes, pool_sizes = None, None, None, None
        if 'n_pools' in kwargs:
            n_pools = kwargs['n_pools']
        if 'n_start_filters' in kwargs:
            n_start_filters = kwargs['n_start_filters']
        if 'kernel_sizes' in kwargs:
            kernel_sizes = kwargs['kernel_sizes']
        if 'pool_sizes' in kwargs:
            pool_sizes = kwargs['pool_sizes']
        return EncoderDecoder(n_pools=n_pools, n_start_filters=n_start_filters,
                              kernel_sizes=kernel_sizes, pool_sizes=pool_sizes)
    else:
        raise ValueError('instance name {} not understood'.format(instance_name))
