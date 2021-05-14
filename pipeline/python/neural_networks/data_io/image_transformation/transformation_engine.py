from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import os
import numpy as np
import cv2
from .transformation_factory import AdvancedTransformationFactory


class TransformationEngine(object):
    __metaclass__ = ABCMeta

    def __init__(self, transformation_params=None,
                 transformation_nesting_orders=None, include_identity=True):
        """
        :param transformation_params:
            { 'rotation': (n, (n floats and / or 'random'))
              'elastic': ()
              'rotation+elastic':
              'linear': (n, (uniform_low_alpha, uniform_high_alpha,
                             uniform_low_beta, uniform_high_beta)
              'poisson': (n, (lambda))
              'gaussian': (n, (mean, variance))
              'linear+gaussian': (n, (uniform_low_alpha, uniform_high_alpha,
                                      uniform_low_beta, uniform_high_beta,
                                      mean, variance)
              'poisson+gaussian': (n, (lambda, mean, variance))
            }
        :param transformation_nesting_orders: list of iterables
        """
        self._transformation_params = transformation_params if transformation_params else {}
        self._transformation_nesting_orders = transformation_nesting_orders
        self._include_identity = include_identity

        self._transformation_pool = []
        self._transformation_id = None
        self._has_transformation = False

        self._factory = AdvancedTransformationFactory()

        self._build_transformation_pool()

    def register_transformation_params(self, transformation_params=None,
                                       transformation_nesting_orders=None,
                                       include_identity=True):
        self._transformation_params = transformation_params if transformation_params else {}
        self._transformation_nesting_orders = transformation_nesting_orders
        self._include_identity = include_identity
        self._build_transformation_pool()

    def empty(self):
        return not self._has_transformation

    @abstractmethod
    def next_transformation(self):
        pass

    def transform(self, images, labels=[], data_format='channels_last'):
        transformation = self.next_transformation()
        return transformation.transform(images=images, labels=labels,
                                        data_format=data_format)

    def _build_transformation_pool(self):
        self._transformation_pool = self._factory.manufacture_queue(
            self._transformation_params,
            nesting_orders=self._transformation_nesting_orders,
            append_identity=self._include_identity)
        self._has_transformation = len(self._transformation_pool) > 0

    def _repr_transformation_pool(self):
        queue_items = []
        for transformation in self._transformation_pool:
            queue_items.append(repr(transformation))
        queue_items = '\n'.join(queue_items)
        return queue_items


class TransformationQueue(TransformationEngine):
    def __init__(self, transformation_params={},
                 transformation_nesting_orders=None, include_identity=True):
        super(TransformationQueue, self).__init__(
            transformation_params=transformation_params,
            transformation_nesting_orders=transformation_nesting_orders,
            include_identity=include_identity)

    def restart(self):
        self._build_transformation_pool()
        self._transformation_id = 0

    def next_transformation(self):
        if self._exhausted():
            self.restart()
        transformation = self._transformation_pool[self._transformation_id]
        self._transformation_id += 1
        return transformation

    def transform(self, images, labels=[], data_format='channels_last'):
        if self._exhausted():
            return None, None
        return super(TransformationQueue, self).transform(images,
                                                          labels=labels,
                                                          data_format=
                                                          data_format)

    # all transformations in pool have been dispensed
    def _exhausted(self):
        return self._transformation_id >= len(self._transformation_pool)

    def __repr__(self):
        output = 'TransformationQueue object transformation_pool:\n'
        return output + self._repr_transformation_pool()


class TransformationRandom(TransformationEngine):
    """
    randomly draw a transformation from pool as next transformation
    uses transform method of super class
    """
    def __init__(self, transformation_params={},
                 transformation_nesting_orders=None, include_identity=True):
        super(TransformationRandom, self).__init__(
            transformation_params=transformation_params,
            transformation_nesting_orders=transformation_nesting_orders,
            include_identity=include_identity)

    def next_transformation(self):
        if not self._has_transformation:
            return None
        transformation_id = np.random.randint(0, len(self._transformation_pool))
        transformation = self._transformation_pool[transformation_id]
        transformation.reset()
        return transformation

    def __repr__(self):
        output = 'TransformationRandom object transformation_pool:\n'
        return output + self._repr_transformation_pool()

    def reset(self):
        for transformation in self._transformation_pool:
            transformation.reset()


def test():
    training_dir = '/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/ground_truths/training'
    img_dir = os.path.join(training_dir, 'image/cluster_msn')
    img_name = 'cluster_msn_151.tif'
    label_dir = os.path.join(training_dir, 'label/cluster_msn')
    label_name = 'cluster_msn_151_label_soma.tif'
    transformation_params = {'rotation': (5, ('random', 'random', 'random', 'random', 'random'))}
    #transformation_params = {'rotation': (5, ('random', 'random', 'random', 'random', 'random')),
                              #'poisson+gaussian': (2, (3, 0, 2))}
    img = cv2.imread(os.path.join(img_dir, img_name), -1)
    label = cv2.imread(os.path.join(label_dir, label_name), 1)[:, :, 2]
    #transformer = TransformationQueue(transformation_params,
                                     #transformation_nesting_orders=[('rotation', 'poisson+gaussian')])
    transformer = TransformationQueue(transformation_params)
    print(transformer)
    print(img.shape)
    i = 0
    imgs = np.zeros(shape=(img.shape[0], img.shape[1], 3), dtype=np.uint16)
    imgs[:, :, 0] = img.copy()
    imgs[:, :, 1] = img.copy()
    imgs[:, :, 2] = img.copy()
    labels = np.zeros(shape=(3000, 3000, 2), dtype=np.uint8)
    labels[:, :, 0] = label[0:3000, 0:3000].copy()
    labels[:, :, 1] = label[0:3000, 0:3000].copy()
    while not transformer.exhausted():
        img_transform, label_transform = transformer.transform(imgs, labels=labels,
                                                               data_format='channels_last')
        img_transform = img_transform.astype(np.float32)
        img_transform *= 255/ np.amax(img_transform)
        img_transform = img_transform.astype(np.uint8)
        print (label_transform.shape)
        assert np.array_equal(label_transform[:, :, 0], label_transform[:, :, 1])
        assert np.array_equal(img_transform[:, :, 0], img_transform[:, :, 1])
        cv2.imwrite('img{}.tif'.format(i), img_transform)
        cv2.imwrite('label{}.tif'.format(i), label_transform[:, :, 0])
        i += 1
    transformer = TransformationQueue()
    print(transformer.exhausted())
    print(transformer)
    transformer.register_transformation_params(transformation_params)
    print(transformer)


if __name__ == '__main__':
    test()
