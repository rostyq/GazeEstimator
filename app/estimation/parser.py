from json import load
from cv2 import imread
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY

from numpy.random import permutation
from numpy import array
from numpy import fliplr
from os import path

from app.estimation.transform import gaze3Dto2D
from app.estimation.transform import pose3Dto2D


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


def get_path_list(path, **kargs):
    return path.format(**kargs).split('/')[::-1]


class DatasetParser:
    """
    Parser for output data from Normalisation module.

    Parameters
    ----------
    images : str
        String with formatter keys `{index}` and {'eye'} that identify path to images data in json.
        Must be like `dataset/{index}/0/eyes/{eye}/`.

    gazes : str
        String with formatter keys `{index}` and {'eye'} that identify path to gazes data in json.
        Must be like `dataset/{index}/0/eyes/{eye}/`.

    poses : str
        String with formatter key `{index}` that identify path to poses vectors data in json.
        Must be like `dataset/{index}/0/`.

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
    >>> gazes = parser.get_gazes_array('left')

    >>> # get array with poses vectors
    >>> poses = parser.get_poses_array()

    >>> # get sample numbers 3 and 4 images of left eyes
    >>> images = list(parser.get_images_array('left', indices=[3, 4]))
    """
    __TEST = ['{eye}', '{index}']
    __EYES = ['left', 'right']

    __FLIP = array([-1, 1, 1])

    def __init__(self, images, gazes, poses):
        """

        """
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
        """
        Reads specific json file to parser.
        Memorizes `path_to_image` files.
        Counts samples in the json data and write to DataserParser.shape.

        Parameters
        ----------
        jsonfile : str
            Path to json file.
        path_to_images : str
            Path to images.

        Returns
        -------
        self : DatasetParser
        """
        self.path_to_images = path_to_images
        self.data = load(jsonfile)
        self.shape = len(self)
        return self

    def __len__(self):
        path_to_samples = self.images.split('{')[0][:-1].split('/')[::-1]
        return len(get_item(self.data, path_to_samples))

    def _check_indices(self, indices):
        if indices is not None:
            max_index = max(indices)
            assert max_index < self.shape - 1, f'Index {max_index} is out of range.'
            return indices
        else:
            return range(self.shape)

    def _check_eye(self, eye):
        assert eye in self.__EYES, 'Wrong eye. There are only `left` and `right`.'

    def get_image(self, index, eye, flip=False, **kwargs):
        """
        Load specific image of an eye.

        Parameters
        ----------
        index : int
            Index of a sample.
        eye : str
            `left` or `right`.
        kwargs : dict
            Key arguments that will pass to `cv2.imread` function.

        Returns
        -------
        image : array-like
        """
        self._check_eye(eye)
        path_to_image = path.join(
            self.path_to_images,
            get_item(
                self.data,
                get_path_list(
                    self.images,
                    index=index,
                    eye=eye
                )
            )
        )
        image = imread(path_to_image, **kwargs)
        if image is None:
            raise Exception(f'Image not found in {path_to_image}')
        else:
            image = cvtColor(image, COLOR_BGR2GRAY)
            if flip:
                return fliplr(image)
            else:
                return image

    def get_pose(self, index, flip=False):
        """
        Returns pose vector of specific sample.

        Parameters
        ----------
        index : int
            Index of a sample.

        Returns
        -------
        pose : list[float, float, float]
        """
        vector = array(get_item(self.data, get_path_list(self.poses, index=index))).reshape(3,)
        if flip:
            return vector * self.__FLIP
        else:
            return vector

    def get_gaze(self, index, eye, flip=False):
        """
        Returns gaze vector of specific sample.

        Parameters
        ----------
        index : int
            Index of a sample.
        eye : str
            `left` or `right`.

        Returns
        -------
        gaze : list[float, float, float]
        """
        self._check_eye(eye)
        vector = array(get_item(self.data, get_path_list(self.gazes, eye=eye, index=index))).reshape(3,)
        if flip:
            return vector * self.__FLIP
        else:
            return vector

    def get_poses_array(self, indices=None, **kwargs):
        """
        Returns batch of pose vectors of samples which number in `indices`.

        Parameters
        ----------
        indices : 1D array-like
            Index of a sample.

        Returns
        -------
        poses : list[list[float, float, float]]
        """
        return [self.get_pose(index, **kwargs) for index in self._check_indices(indices)]

    def get_gazes_array(self, eye, indices=None, **kwargs):
        """
        Returns batch of gaze vectors of samples which number in `indices`.

        Parameters
        ----------
        indices : 1D array-like
            Index of a sample.
        eye : str
            `left` or `right`.

        Returns
        -------
        gazes : list[list[float, float, float]]
        """
        self._check_eye(eye)
        return [self.get_gaze(index, eye=eye, **kwargs) for index in self._check_indices(indices)]

    def get_images_array(self, eye, indices=None, **kwargs):
        """
        Returns batch of images of samples which number in `indices`.

        Parameters
        ----------
        indices : 1D array-like
            Index of a sample.
        eye : str
            `left` or `right`.

        Returns
        -------
        gazes : list[array-like]
        """
        self._check_eye(eye)
        return [self.get_image(index, eye, **kwargs) for index in self._check_indices(indices)]

    def get_full_data(self, indices=None):

        eyes, poses, gazes = [], [], []

        for flip, eye in enumerate(self.__EYES):

            eyes.extend(self.get_images_array(eye=eye, flip=bool(flip), indices=indices))
            poses.extend(self.get_poses_array(indices=indices, flip=bool(flip)))
            gazes.extend(self.get_gazes_array(eye=eye, indices=indices, flip=bool(flip)))

        eyes = array(eyes, subok=True).reshape(-1, 36, 60, 1) / 255
        poses = pose3Dto2D(array(poses, subok=True))
        gazes = gaze3Dto2D(array(gazes, subok=True))
        return eyes, poses, gazes
