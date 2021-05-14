import os
import h5py
import numpy as np
import cv2


def ims_data_shape(ims_file_path, channel=0, resolution_level=0):
    """
    :param ims_file_path:
    :param resolution_level:
    :param channel:
    :return: shape of data at given resolution level and channel
    """
    if not os.path.isfile(ims_file_path):
        raise ValueError('{} does not exist'.format(ims_file_path))
    ims = h5py.File(ims_file_path, 'r')
    try:
        resolution_level_datasets = ims['DataSet/ResolutionLevel {}/TimePoint 0'.format(resolution_level)]
    except KeyError:
        raise KeyError('requested resolution level {} does not exist'
                       .format(resolution_level))
    try:
        channel_data_set = resolution_level_datasets['Channel {}/Data'.format(channel)]
    except KeyError:
        raise KeyError('requested channel {} does not exist'
                       .format(channel))
    return channel_data_set.shape


def ims_n_channels(ims_file_path, resolution_level=0):
    if not os.path.isfile(ims_file_path):
        raise ValueError('{} does not exist'.format(ims_file_path))
    ims = h5py.File(ims_file_path, 'r')
    try:
        multi_channel_datasets = ims['DataSet/ResolutionLevel {}/TimePoint 0'.format(resolution_level)]
    except KeyError:
        raise KeyError('requested resolution level {} does not exist'.format(resolution_level))
    return len(multi_channel_datasets.keys())


def read_ims(ims_file_path, zyx_offsets, zyx_extents=(-1, -1, -1), channels=(0,), resolution_level=0):
    """
    return subvolume(s) of data defined by zyx_offsets and zyx_extents, at
    requested resolution_level and channel(s). this function assumes only
    time point 0 exists
    :param zyx_offsets: tuple of 3 integers
    :param zyx_extents: if negative, select till end of axis
    :param channels: list of channel ids
    :return: numpy array (or list of numpy arrays of multiple channels requested)
    """
    if not os.path.isfile(ims_file_path):
        raise ValueError('{} does not exist'.format(ims_file_path))
    if len(zyx_offsets) != 3 or len(zyx_extents) != 3:
        raise ValueError('zyx_offsets and zyx_extents should be of length 3')
    ims = h5py.File(ims_file_path, 'r')
    try:
        multi_channel_datasets = ims['DataSet/ResolutionLevel {}/TimePoint 0'.format(resolution_level)]
    except KeyError:
        raise KeyError('requested resolution level {} does not exist'.format(resolution_level))
    for channel in channels:
        if channel < 0 or channel >= len(multi_channel_datasets.keys()):
            raise ValueError('{} is not a valid channel id'.format(channel))
    data = []
    for channel in channels:
        channel_dataset = multi_channel_datasets['Channel {}/Data'.format(channel)]
        z_start, y_start, x_start = zyx_offsets
        z_stop = zyx_extents[0] + z_start if zyx_extents[0] > 0 else channel_dataset.shape[0]
        y_stop = zyx_extents[1] + y_start if zyx_extents[1] > 0 else channel_dataset.shape[1]
        x_stop = zyx_extents[2] + x_start if zyx_extents[2] > 0 else channel_dataset.shape[2]
        if z_start < 0 or y_start < 0 or x_start < 0:
            raise ValueError('offset values must be non negative')
        if z_stop - z_start > channel_dataset.shape[0] or y_stop - y_start > channel_dataset.shape[1] or \
           x_stop - x_start > channel_dataset.shape[2]:
            raise ('extent values out of range')
        # skipping range checking and assume 16bit. quick script
        img = np.zeros(shape=(z_stop - z_start, y_stop - y_start, x_stop - x_start), dtype=np.uint16)
        channel_dataset.read_direct(img, source_sel=(np.s_[z_start: z_stop, y_start: y_stop, x_start: x_stop]))
        data.append(img.copy())
    ims.close()
    return data[0] if len(data) == 1 else data


def imaris_to_tif_sequence(imaris_path, channel=0, resolution_level=0, volume_name=None, write_resolution_dir=False):
    """
    convert imaris file to tif sequence at the channel and resolution level
    requested. imaris_path should be full path to the imaris file.
    if the imaris file is located in /path/to/imaris_dir, with channel 0 and
    resolution_level 0 requested, output is generated to
    /path/to/imaris_dir/[imaris_file_basename]_ch0_tif/[resolution0]
    """
    assert os.path.isfile(imaris_path)
    if not volume_name:
        volume_name = os.path.basename(imaris_path).replace('.ims', '')
    out_dir = os.path.join(os.path.dirname(imaris_path), '{}_ch{}_tif'.format(volume_name, channel))
    if write_resolution_dir:
        out_dir = os.path.join(out_dir, 'resolution{}'.format(resolution_level))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    data_shape = ims_data_shape(imaris_path, resolution_level=resolution_level, channel=channel)
    z_levels = data_shape[0]
    for z in range(0, z_levels, 8):
        try:
            # 8 is a common z chunk size. reading 8 z planes at a time avoid
            # redundant chunk read
            image = read_ims(imaris_path, (z, 0, 0), zyx_extents=(min(8, z_levels - z), -1, -1),
                             channels=[channel], resolution_level=resolution_level)
            image = np.squeeze(image)
            for z_ in range(z, min(z + 8, z_levels)):
                z_str = str(z_).zfill(len(str(z_levels)))
                image_name = '{}_Z{}.tif'.format(volume_name, z_str)
                cv2.imwrite(os.path.join(out_dir, image_name), image[z_ - z, :, :])
                print(image_name)
        except IOError as e:
            print('can not read z = {}: {}'.format(z, str(e)))


if __name__ == '__main__':
    imaris_to_tif_sequence('/media/muyezhu/Mumu/morph_project/raw_images/etv1/Etv1-MORF3_MS05-3_30x_Ctx_c05A_crop01/Etv1-MORF3_MS05-3_30x_Ctx_c05A_crop01_FusionStitcher.ims')
