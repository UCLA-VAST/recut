from __future__ import print_function, division
from itertools import product, chain
from .fundamental_transformations import Transformation, Identity, Rotation, \
     Reverse, LinearTransform, ElasticTransform, GaussianNoise, PoissonNoise


class PipelineTransformation(Transformation):
    """
    holds a sequence of fundamental or pipeline augmentation instances
    apply() method calls the apply() method of the instances and pipelines
    the input images through
    e.g. (Rotation, Gaussian, Elastic)
         img -> Rotation -> rotated image -> Gaussian ->
         rotated image + gaussian noise -> Elastic ->
         rotated image + gaussian noise + elastic transform
    """
    def __init__(self, component_transformations):
        for transformation in component_transformations:
            if not isinstance(transformation, Transformation):
                raise TypeError('elements of component_augmentations must be '
                                'Augmentation instances')
        self.component_transformations = component_transformations

    def __repr__(self):
        repr_str = 'Pipelinetransformations = '
        obj_strs = []
        for transformation in self.component_transformations:
            obj_strs.append(repr(transformation))
        obj_str = ' + '.join(obj_strs)
        repr_str += obj_str
        return repr_str

    def transform_2d(self, input_img, is_label):
        for component_transformation in self.component_transformations:
            input_img = component_transformation.transform_2d(input_img, is_label)
        return input_img

    def transform_nd(self, input_img, is_label, data_format='channels_last'):
        for transformation in self.component_transformations:
            input_img = transformation.transform_nd(input_img, is_label,
                                                    data_format=data_format)
        return input_img

    def reset(self):
        for transformation in self.component_transformations:
            transformation.reset()


class RotationAndElasticTransform(PipelineTransformation):
    def __init__(self):
        super(RotationAndElasticTransform, self).__init__()
        pass


class PoissonAndGaussianNoise(PipelineTransformation):
    def __init__(self, poisson_lambda=3, mean=0, variance=1):
        factory = TransformationFactory()
        super(PoissonAndGaussianNoise, self).__init__(
            [factory.manufacture('poisson', poisson_lambda),
             factory.manufacture('gaussian', mean, variance)])

        self.poisson_lambda = poisson_lambda
        self.mean = mean
        self.variance = variance

    def __repr__(self):
        return 'PoissonAndGaussianNoise({}, {}, {})' \
               .format(self.poisson_lambda, self.mean, self.variance)


class TransformationFactory:
    """
    create single augmentation instance defined in recipe
    """
    recipe = {'rotation': Rotation, 'elastic': ElasticTransform,
              'rotation+elastic': RotationAndElasticTransform,
              'reverse': Reverse,
              'poisson': PoissonNoise, 'gaussian': GaussianNoise,
              'poisson+gaussian': PoissonAndGaussianNoise, 'identity': Identity}

    def __init__(self):
        self.transformation_name = None

    @staticmethod
    def _confirm_supported(transformation_name):
        if transformation_name not in TransformationFactory.recipe:
            raise ValueError('{} is not a supported transformation operation'
                             .format(transformation_name))

    def manufacture(self, transformation_name, *transformation_args):
        TransformationFactory._confirm_supported(transformation_name)
        self.transformation_name = transformation_name
        return TransformationFactory.recipe[self.transformation_name](*transformation_args)


class AdvancedTransformationFactory:
    def __init__(self):
        self.factory = TransformationFactory()

    def manufacture_pipeline(self, transformation_names_args=None,
                             component_transformations=None):
        """
        :param transformation_names_args: [(augmentation_name, args_tuple)]
                                        list of tuple, each tuple has a
                                        augmentation name and tuple with its
                                        constructor arguments
        :param component_transformations: component augmentations to be made into
                                        complex augmentation
        :return: ComplexAugmentation instance
        """
        if (transformation_names_args is None) == \
                (component_transformations is None):
            raise ValueError('one and only one keyword arguments '
                             'augmentation_names_args and simple_augmentations '
                             'should be not None')
        if transformation_names_args is not None:
            component_transformations = []
            for transformation_name_args in transformation_names_args:
                transformation_name, transformation_args = transformation_name_args
                if not isinstance(transformation_name, str) or \
                        not isinstance(transformation_args, tuple):
                    raise ValueError('augmentation_name_args should be in the '
                                     'form tuple(str, tuple)')
                component_transformations.append(self.factory
                                                 .manufacture(transformation_name,
                                                              *transformation_args))
        else:
            for component_transformation in component_transformations:
                if not isinstance(component_transformation, Transformation):
                    raise TypeError('component_augmentations items must all '
                                    'be Augmentation instances')
        return PipelineTransformation(component_transformations)

    def manufacture_multiple(self, transformation_name, n_transformation_args):
        """
        :param transformation_name
        :param n_transformation_args: tuple. one of the value format in
                                    augmentation_params of manufacture_queue
        :return: return a list of n Augmentation instances of the same type
                 (determined by augmentation_name)
        """
        n_transformation = []
        n, transformation_args = n_transformation_args
        if n < 0:
            raise ValueError('number of Augmentation instances requested '
                             'must be non negative')
        if transformation_name == 'rotation':
            for theta in transformation_args:
                n_transformation.append(
                    self.factory.manufacture(transformation_name, theta))
        else:
            for i in range(n):
                n_transformation.append(
                    self.factory.manufacture(transformation_name,
                                             *transformation_args))
        return n_transformation

    def _permutate(self, nesting_transformations):
        """
        permutate augmentations and create pipeline augmentations
        nesting_augmentations are a list of lists, each lists consist of
        augmentation of the same type
        :param nesting_transformations:
        :param nesting_order:
        :return:
        """
        queue = []
        for component_augmentations in product(*nesting_transformations):
            queue.append(self.manufacture_pipeline(
                component_transformations=component_augmentations))
        return queue

    def manufacture_queue(self, transformation_params, nesting_orders=None,
                          append_identity=True):
        """
        construct a queue of augmentations
        :param transformation_params:
                 { 'rotation': (n, (n floats and / or 'random'))
                   'elastic': ()
                   'rotation+elastic':
                   'poisson': (n, (lambda))
                   'gaussian': (n, (mean, variance))
                   'poisson+gaussian': (n, (lambda, mean, variance))  }
        :param nesting_orders: permutation order to create pipeline
               transformation from simpler transformations
               e.g. nesting_orders = ((rotation, poisson, reverse),
               (rotation, poisson)) creates pipeline augmentations
               pipeline(rotation0, poisson0, reverse),
               pipeline(rotation0, poisson1, reverse),
               pipeline(rotation1, poisson0, reverse),
               pipeline(rotation1, poisson1, reverse),
               pipeline(rotation0, poisson0),
               pipeline(rotation0, poisson1),
               pipeline(rotation1, poisson0),
               pipeline(rotation1, poisson1),
               augmentations not included in nesting_order will simply be
               constructed and appended to queue
        :param append_identity: if True, append Identity transformation at end
               of queue
        :return:
        """
        queue = []
        if nesting_orders is not None:
            for nesting_order in nesting_orders:
                nesting_transformations = []
                for nesting_transformation_name in nesting_order:
                    assert nesting_transformation_name in transformation_params
                    n_augmentation_args = transformation_params[nesting_transformation_name]
                    nesting_transformations.append(
                        self.manufacture_multiple(nesting_transformation_name,
                                                  n_augmentation_args))
                queue.extend(self._permutate(nesting_transformations))
        if nesting_orders is None:
            nesting_orders = []
        nesting_orders = set(chain(*nesting_orders))
        for transformation_name in transformation_params:
            if transformation_name not in nesting_orders:
                n_augmentation_args = transformation_params[transformation_name]
                queue.extend(self.manufacture_multiple(transformation_name,
                                                       n_augmentation_args))
        if append_identity:
            AdvancedTransformationFactory.append_identity(queue)
        return queue

    @staticmethod
    def append_identity(transformation_queue):
        for transformation in transformation_queue:
            if not isinstance(transformation, Transformation):
                raise TypeError('queue items must be all be Transformation instances')
        transformation_queue.append(Identity())


def test():
    factory = AdvancedTransformationFactory()
    transformations = factory.manufacture_pipeline(transformation_names_args=
                                                   [('rotation', ('random',)),
                                                    ('poisson', (1,)),
                                                    ('gaussian', (0, 1)),
                                                    ('poisson+gaussian', (1, 0, 1))])
    print(transformations)
    transformations = factory.manufacture_multiple('rotation', (5, (3, 3, 1, 'random', 'random')))
    print(transformations)
    transformations = factory.manufacture_queue({'rotation': (5, (3, 3, 1, 'random', 'random')),
                                                 'gaussian': (3, (0, 1)),
                                                 'poisson+gaussian': (2, (1, 0, 1))})
    print(transformations)
    transformations = factory.manufacture_queue({'rotation': (5, (3, 3, 1, 'random', 'random')),
                                                 'gaussian': (3, (0, 1)),
                                                 'poisson+gaussian': (2, (1, 0, 1))})
    print(transformations)
    transformations = factory.manufacture_queue({'rotation': (5, (3, 2, 1, 'random', 'random')),
                                                 'poisson+gaussian': (2, (1, 0, 1)),
                                                 'reverse': (1, ())},
                                                nesting_orders=[('rotation', 'reverse'),
                                                                ('rotation', ),
                                                                ('rotation', 'poisson+gaussian', 'reverse'),
                                                                ('rotation', 'poisson+gaussian')],
                                                append_identity=True)
    print(transformations)


if __name__ == '__main__':
    test()