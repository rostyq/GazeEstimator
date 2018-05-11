from json import load
from cv2 import imread


def get_item(data: dict, path_list: list):
    key = path_list.pop()
    try:
        key = int(key)
    except ValueError:
        pass
    if not path_list:
        return data.__getitem__(key)
    else:
        return get_item(data.__getitem__(key), path_list)


class DatasetParser:
    """
    Examples
    --------

    >>> # Instanciate parser with information about jsons structure:
    >>> faces_path = 'dataset/{index}/0/'
    >>> eyes_path = faces_path+'eyes/{eye}/'
    >>> parser = DatasetParser(images=eyes_path+'image', gazes=eyes_path+'gaze', poses=faces_path+'rotation')

    >>> # Fit json to parser:
    >>> DATASET_PATH = '/path/to/folder/with/images_and_json'
    >>> JSON_NAME = 'dataset.json'
    >>> with open(DATASET_PATH+JSON_NAME, 'r') as file:
    ...     parser.fit(file, DATASET_PATH)

    >>> # get array with gazes vectors
    >>> gazes = list(parser.get_gazes_array('left'))

    >>> # get array with poses vectors
    >>> poses = list(parser.get_poses_array())

    >>> # get sample numbers 3 and 4 images of left eyes
    >>> images = list(parser.get_images_array('left', indices=[3, 4]))
    """
    __TEST = ['{eye}', '{index}']
    __EYES = ['left', 'right']

    def __init__(self, images, gazes, poses):

        if any([(not frmt in check) for frmt, check in zip(self.__TEST,
                                                           [''.join([images, gazes]), poses])]):
            raise Exception('Wrong formating keys in paths.')

        self.images = images
        self.gazes = gazes
        self.poses = poses
        self.data = None
        self.shape = None
        self.path_to_images = None

    def fit(self, jsonfile, path_to_images):
        self.path_to_images = path_to_images
        self.data = load(jsonfile)
        self.shape = len(self)
        return self

    def __len__(self):
        path_to_samples = self.images.split('{')[0][:-1].split('/')[::-1]
        return len(get_item(self.data, path_to_samples))

    @staticmethod
    def _get_path_list(path, **kargs):
        return path.format(**kargs).split('/')[::-1]

    def _check_indices(self, indices):
        if indices is not None:
            max_index = max(indices)
            assert max_index < self.shape - 1, f'Index {max_index} is out of range.'
            return indices
        else:
            return range(self.shape)

    def _check_eye(self, eye):
        assert eye in self.__EYES, 'Wrong eye. There are only `left` and `right`.'

    def get_image(self, index, eye, **kwargs):
        self._check_eye(eye)
        path_to_image = self.path_to_images+get_item(
            self.data,
            self._get_path_list(
                self.images,
                index=index,
                eye=eye
            )
        )
        image = imread(path_to_image, **kwargs)
        if image is None:
            raise Exception(f'Image not found in {path_to_image}')
        else:
            return image

    def get_pose(self, index):
        return get_item(self.data, self._get_path_list(self.poses, index=index))

    def get_gaze(self, index, eye):
        self._check_eye(eye)
        return get_item(self.data, self._get_path_list(self.gazes, eye=eye, index=index))

    def get_poses_array(self, indices=None):
        return [self.get_pose(index) for index in self._check_indices(indices)]

    def get_gazes_array(self, eye, indices=None):
        self._check_eye(eye)
        return [self.get_pose(index) for index in self._check_indices(indices)]

    def get_images_array(self, eye, indices=None, **kwargs):
        self._check_eye(eye)
        return [self.get_image(index, eye, **kwargs) for index in self._check_indices(indices)]
