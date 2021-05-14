from pprint import pformat
import numpy as np
import tensorflow as tf


class LabelRules:
    def __init__(self, label_input_colors, label_weights={}):
        """
        define interpretation of labeled images
        :param label_input_colors: e.g. {'soma': (r, g, b),
                                         'background': (0, 0, 0)}
        :param label_weights: e.g. {'soma': 10, 'background': 1}
        """
        if len(label_input_colors) == 0:
            raise ValueError('label_input_colors must be non empty')
        self.input_colors = label_input_colors
        self.rgb_input = False
        self.output_colors = {}
        self.input_to_output_colors = {}
        self.output_to_input_colors = {}
        output_foreground_color = 1
        for label in self.input_colors:
            if not isinstance(self.input_colors[label], tuple):
                raise TypeError('label input colors must be expressed as '
                                'tuples, e.g. (255,) for gray, '
                                '(255, 255, 0) for rgb')
            if label == 'background':
                self.output_colors[label] = 0
                self.output_to_input_colors[0] = self.input_colors[label]
                self.input_to_output_colors[self.input_colors[label]] = 0
            else:
                self.output_colors[label] = output_foreground_color
                self.output_to_input_colors[output_foreground_color] = self.input_colors[label]
                self.input_to_output_colors[self.input_colors[label]] = output_foreground_color
                output_foreground_color += 1
            if len(self.input_colors[label]) == 3:
                self.rgb_input = True
        # output_color: weight
        self.label_weights = {}
        for label in self.output_colors:
            output_color = self.output_colors[label]
            if label not in label_weights:
                self.label_weights[output_color] = 1
            else:
                self.label_weights[output_color] = label_weights[label]

    def __repr__(self):
        info = '{}\n{}\n'.format(self.__class__.__name__,
                                 pformat(self.output_colors))
        return info

    def n_labels(self):
        return len(self.input_colors)

    def label_weights_sorted_by_color(self):
        weights = np.ones(shape=(len(self.input_colors),),
                          dtype=np.float32)
        for color in self.output_colors.values():
            weights[color] = self.label_weights[color]
        return weights

    def translate_to_output(self, label_image):
        if not (len(label_image.shape) / 3 == 1) == self.rgb_input:
            raise ValueError('LabelRule instance and label must be both rgb or both gray')
        if isinstance(label_image, tf.Tensor):
            label_image = label_image.numpy()

        if self.rgb_input:
            label_image = label_image.astype(np.uint32)
            label_image = np.left_shift(label_image[:, :, 2], 16) + \
                          np.left_shift(label_image[:, :, 1], 8) + \
                          label_image[:, :, 0]
        for input_color, output_color in self.input_to_output_colors.items():
            if self.rgb_input:
                input_color = np.left_shift(input_color[0], 16) + \
                              np.left_shift(input_color[1], 8) + input_color[2]
            label_image[label_image == input_color] = output_color
        label_image = label_image.astype(np.int32)
        assert set(np.unique(label_image)).issubset(
            set(self.output_colors.values())), \
            'label_image is not consistent with LabelRule instance'
        return label_image

    def translate_to_input(self, predicted_image):
        assert len(predicted_image.shape) == 2, 'predicted_image must be 2d'
        assert predicted_image.dtype in (np.int32, tf.int32), \
            'predicted_image must be int32, got {} instead'.format(predicted_image.dtype)
        if isinstance(predicted_image, tf.Tensor):
            predicted_image = predicted_image.numpy()
        if self.rgb_input:
            result = np.zeros(shape=(predicted_image.shape[0],
                                     predicted_image.shape[1], 3),
                              dtype=np.uint8)
        else:
            result = np.zeros(shape=predicted_image.shape, dtype=np.uint8)
        for output_color, input_color in self.output_to_input_colors.items():
            if self.rgb_input:
                r, g, b = input_color
                result[:, :, 2][predicted_image == output_color] = r
                result[:, :, 1][predicted_image == output_color] = g
                result[:, :, 0][predicted_image == output_color] = b
            else:
                result[predicted_image == output_color] = input_color
        return result


dummy_label_rules = LabelRules({'0': (0,), '1': (1,)})
