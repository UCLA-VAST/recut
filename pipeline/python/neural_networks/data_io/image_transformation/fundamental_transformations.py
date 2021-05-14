from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.random import uniform, poisson, normal
import cv2


class Transformation:
    __metaclass__ = ABCMeta

    @abstractmethod
    def transform_2d(self, input_img, is_label, **kwargs):
        pass

    @abstractmethod
    def transform_nd(self, input_img, is_label,
                     data_format='channels_last', **kwargs):
        pass

    @abstractmethod
    # for instances with randomly drawn attributes during construction, redraw
    # the attributes
    def reset(self):
        pass

    def _apply_array(self, input_img, is_label,
                     data_format='channels_last', **kwargs):
        output_img = input_img.copy()
        if output_img.ndim == 2:
            return self.transform_2d(output_img, is_label, **kwargs)
        else:
            return self.transform_nd(output_img, is_label,
                                     data_format=data_format, **kwargs)

    def _transform_nd_with_2d(self, input_img, is_label,
                              data_format='channels_last', **kwargs):
        if input_img.ndim == 2:
            return self.transform_2d(input_img, is_label)

        elif input_img.ndim == 3:
            if data_format == 'channels_first':
                n_channels = input_img.shape[0]
                for i in range(n_channels):
                    input_img[i, :, :] = self.transform_2d(input_img[i, :, :],
                                                           is_label, **kwargs)
                return input_img
            elif data_format == 'channels_last':
                n_channels = input_img.shape[-1]
                for i in range(n_channels):
                    input_img[:, :, i] = self.transform_2d(input_img[:, :, i],
                                                           is_label, **kwargs)
                return input_img
            else:
                raise ValueError('unknown data format')

        elif input_img.ndim == 4:
            batch_size = input_img.shape[0]
            if data_format == 'channels_first':
                n_channels = input_img.shape[1]
                for i in range(batch_size):
                    for channel in range(n_channels):
                        input_img[i, channel, :, :] = \
                            self.transform_2d(input_img[i, channel, :, :],
                                              is_label, **kwargs)
                return input_img
            elif data_format == 'channels_last':
                n_channels = input_img.shape[-1]
                for i in range(batch_size):
                    for channel in range(n_channels):
                        input_img[i, :, :, channel] = \
                            self.transform_2d(input_img[i, :, :, channel],
                                              is_label, **kwargs)
                return input_img
            else:
                raise ValueError('unknown data format')
        else:
            raise ValueError('unsupported array ndim')

    def _transform_list(self, input_imgs, is_label,
                        data_format='channels_last', **kwargs):
        output_images = []
        for input_img in input_imgs:
            assert isinstance(input_img, np.ndarray)
            output_images.append(self._apply_array(input_img, is_label,
                                                   data_format=data_format,
                                                   **kwargs))
        return output_images

    def transform(self, images=[], labels=[],
                  data_format='channels_last', **kwargs):
        if isinstance(images, np.ndarray):
            output_images = self._apply_array(images, False,
                                              data_format=data_format, **kwargs)
        else:
            if len(images) > 0:
                output_images = self._transform_list(images, False,
                                                     data_format=data_format,
                                                     **kwargs)
            else:
                output_images = None
        if isinstance(labels, np.ndarray):
            output_labels = self._apply_array(labels, True,
                                              data_format=data_format, **kwargs)
        else:
            if len(labels) > 0:
                output_labels = self._transform_list(labels, True,
                                                     data_format=data_format,
                                                     **kwargs)
            else:
                output_labels = None
        if output_images is not None and output_labels is not None:
            return output_images, output_labels
        elif output_images is not None:
            return output_images
        elif output_labels is not None:
            return output_labels
        else:
            return


class Identity(Transformation):
    def __init__(self):
        pass

    def __repr__(self):
        return 'Identity'

    def transform_2d(self, input_img, is_label, **kwargs):
        return input_img

    def transform_nd(self, input_img, is_label,
                     data_format='channels_last', **kwargs):
        return input_img

    def reset(self):
        pass


class Reverse(Transformation):
    def __init__(self):
        self.n_scales = 1
        self.n_contexts = None

    def __repr__(self):
        return 'Reverse image sequence'

    def transform_2d(self, input_img, is_label, **kwargs):
        return input_img

    def transform_nd(self, input_img, is_label,
                     data_format='channels_last', **kwargs):
        if 'n_scales' in kwargs:
            self.n_scales = kwargs['n_scales']
        if input_img.ndim == 2:
            return input_img
        elif input_img.ndim == 3:
            if data_format == 'channels_first':
                assert input_img.shape[0] % self.n_scales == 0
                self.n_contexts = input_img.shape[0] // self.n_scales
                for i in range(self.n_scales):
                    input_img[i * self.n_scales:
                              i * self.n_scales + self.n_contexts] = \
                        np.flip(input_img[i * self.n_scales:
                                          i * self.n_scales + self.n_contexts],
                                0)
                return input_img
            elif data_format == 'channels_last':
                assert input_img.shape[2] % self.n_scales == 0
                self.n_contexts = input_img.shape[2] // self.n_scales
                for i in range(self.n_scales):
                    input_img[i * self.n_scales:
                              i * self.n_scales + self.n_contexts] = \
                        np.flip(input_img[i * self.n_scales:
                                          i * self.n_scales + self.n_contexts],
                                -1)
                return input_img
            else:
                raise ValueError('unknown data format')
        elif input_img.ndim == 4:
            if data_format == 'channels_first':
                assert input_img.shape[1] % self.n_scales == 0
                self.n_contexts = input_img.shape[1] // self.n_scales
                for i in range(self.n_scales):
                    input_img[:, i * self.n_scales:
                              i * self.n_scales + self.n_contexts] = \
                        np.flip(input_img[:, i * self.n_scales:
                                          i * self.n_scales + self.n_contexts],
                                1)
                return input_img
            elif data_format == 'channels_last':
                assert input_img.shape[3] % self.n_scales == 0
                self.n_contexts = input_img.shape[3] // self.n_scales
                for i in range(self.n_scales):
                    input_img[..., i * self.n_scales:
                              i * self.n_scales + self.n_contexts] = \
                        np.flip(input_img[..., i * self.n_scales:
                                          i * self.n_scales + self.n_contexts],
                                -1)
                return input_img
            else:
                raise ValueError('unknown data format')
        else:
            raise ValueError('unsupported array ndim')

    def reset(self):
        pass


class Rotation(Transformation):
    def __init__(self, theta):
        """
        :param theta: angle of clock wise rotation
        """
        self.random_theta = theta == 'random'
        if theta == 'random':
            theta = Rotation._init_random_theta()
        elif not 0 <= theta <= 360:
            raise ValueError('theta must be a value in range [0, 360]')
        self.theta = theta

    def __repr__(self):
        return 'Rotation({})'.format(self.theta)

    @staticmethod
    def _init_random_theta():
        return uniform(low=0.0, high=360.0)

    def transform_2d(self, input_img, is_label, **kwargs):
        if self.theta == 0 or self.theta == 360:
            return input_img
        # use rot90 only if theta is multiples of 90 and input image is square
        elif self.theta % 90 == 0 and input_img.shape[0] == input_img.shape[1]:
            k = self.theta // 90
            # axes = (1, 0) produce clock wise rotation
            return np.rot90(input_img, k=k, axes=(1, 0))
        else:
            m = cv2.getRotationMatrix2D((input_img.shape[1] // 2,
                                         input_img.shape[0] // 2),
                                        -self.theta, 1)
            # label uses INTER_NEAREST to avoid generate uninterpretable
            # intensity values
            flag = cv2.INTER_LINEAR if not is_label else cv2.INTER_NEAREST
            return cv2.warpAffine(input_img, m, (input_img.shape[1],
                                                 input_img.shape[0]),
                                  flags=flag,
                                  borderMode=cv2.BORDER_CONSTANT)

    def transform_nd(self, input_img, is_label,
                     data_format='channels_last', **kwargs):
        return self._transform_nd_with_2d(input_img, is_label,
                                          data_format=data_format, **kwargs)

    def reset(self):
        if self.random_theta:
            self.theta = Rotation._init_random_theta()


class ElasticTransform(Transformation):
    def __init__(self):
        pass

    def transform_2d(self, input_img, is_label, **kwargs):
        pass

    def transform_nd(self, input_img, is_label,
                     data_format='channels_last', **kwargs):
        pass

    def reset(self):
        pass


class LinearTransform(Transformation):
    def __init__(self, alpha_low=0.5, alpha_high=2,
                 beta_low=None, beta_high=None):
        if alpha_low >= alpha_high:
            raise ValueError('alpha low much be smaller than alpha high')
        if beta_low is not None and beta_high is not None:
            if beta_low >= beta_high:
                raise ValueError('beta low much be smaller than beta high')
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.beta_low = beta_low
        self.beta_high = beta_high
        self._init_alpha_beta()

    def _init_alpha_beta(self):
        self.alpha = uniform(low=self.alpha_low, high=self.alpha_high)
        if self.beta_low is None or self.beta_high is None:
            self.beta = None
        else:
            self.beta = uniform(low=self.beta_low, high=self.beta_high)

    def transform_2d(self, input_img, is_label, **kwargs):
        return self.transform_2d(input_img, is_label)

    def transform_nd(self, input_img, is_label,
                     data_format='channels_last', **kwargs):
        if self.beta is None:
            intensity_range = np.amax(input_img) - np.amin(input_img)
            self.beta = uniform(low=-intensity_range / 10,
                                high=intensity_range / 10)
        if is_label:
            return input_img

        input_dtype = input_img.dtype
        input_img = input_img.astype(np.float32)
        input_img *= self.alpha
        input_img += self.beta
        input_img = np.clip(input_img, 0, np.iinfo(input_dtype).max)
        return input_img.astype(input_dtype)

    def reset(self):
        self._init_alpha_beta()


class GaussianNoise(Transformation):
    def __init__(self, mean=0, variance=1):
        if variance <= 0:
            raise ValueError('gaussian distribution variance '
                             'parameter must be positive')
        self.mean = mean
        self.variance = variance

    def __repr__(self):
        return 'GaussianNoise({}, {})'.format(self.mean, self.variance)

    def transform_2d(self, input_img, is_label, **kwargs):
        return self.transform_nd(input_img, is_label)

    def transform_nd(self, input_img, is_label,
                     data_format='channels_last', **kwargs):
        if is_label:
            return input_img
        gaussian_noise = normal(self.mean, np.sqrt(self.variance),
                                size=input_img.shape)
        img_dtype = input_img.dtype
        input_img = input_img.astype(np.float32)
        input_img += gaussian_noise
        input_img = np.clip(input_img, 0, np.iinfo(img_dtype).max)
        return input_img.astype(img_dtype)

    def reset(self):
        pass


class PoissonNoise(Transformation):
    def __init__(self, poisson_lambda=3):
        if poisson_lambda <= 0:
            raise ValueError('poisson distribution lambda '
                             'parameter must be positive')
        self.poisson_lambda = poisson_lambda

    def __repr__(self):
        return 'PoissonNoise({})'.format(self.poisson_lambda)

    def transform_2d(self, input_img, is_label, **kwargs):
        return self.transform_nd(input_img, is_label, **kwargs)

    def transform_nd(self, input_img, is_label,
                     data_format='channels_last', **kwargs):
        if is_label:
            return input_img
        poisson_noise = poisson(self.poisson_lambda, size=input_img.shape)
        img_dtype = input_img.dtype
        input_img = input_img.astype(np.float32)
        input_img += poisson_noise
        input_img = np.clip(input_img, 0, np.iinfo(img_dtype).max)
        return input_img.astype(img_dtype)

    def reset(self):
        pass


