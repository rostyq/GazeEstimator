from app.device import SceneObj
from numpy import cross
from numpy import array
from app.parser import quaternion_to_angle_axis, face_point_to_array

class Actor(SceneObj):


    def __init__(self, name, origin, rectangle=None, landmarks2D=None):
        super().__init__(name=name, origin=origin)
        self.rectangle = rectangle
        self.landmarks3D = {}
        self.landmarks2D = landmarks2D
        self.gazes = None

    def __dict__(self):
        return {
            'eyes': {
                'left': {
                    'landmarks': self.landmarks.tolist()[4:],
                    'center': self.left_eye_center.tolist(),
                    'gaze': self.real_left_gaze.tolist(),
                    'gaze_norm': self.real_left_gaze_norm_camera.tolist(),
                    'image': None
                },
                'right': {
                    'landmarks': self.landmarks.tolist()[:4],
                    'center': self.right_eye_center.tolist(),
                    'gaze': self.real_right_gaze.tolist(),
                    'gaze_norm': self.real_right_gaze_norm_camera.tolist(),
                    'image': None,
                }
            },
            'rotation': self.rotation_vector.tolist(),
            'rotation_norm': self.rotation_vector_norm_camera.tolist(),
            'translation': self.translation_vector.tolist() if self.translation_vector is not None else None,
        }

    def set_landmarks3d(self, face_points):
        left_eye_rect = array([face_points[i] for i in [1080, 201, 289, 151]])
        right_eye_rect = array([face_points[i] for i in [1084, 847, 947, 772]])

        LeyeO = array(face_points[469])
        LeyeI = array(face_points[210])
        ReyeO = array(face_points[469])
        ReyeI = array(face_points[210])
        nose = array(face_points[18])
        chin = array(face_points[4])

        self.landmarks3D = {
            'eyes': {
                'left': {
                    'rectangle': left_eye_rect,
                    'center': (LeyeO + LeyeI)/2
                },
                'right': {
                    'rectangle': right_eye_rect,
                    'center': (ReyeO + ReyeI)/2
                }
            },
            'nose': nose,
            'chin': chin
        }
        return self

    def set_landmarks2d(self, shape_from_dlib):
        self.landmarks2D = shape_from_dlib
        return self

    def get_eye_landmarks2d(self):
        return self.landmarks2D[[37, 40] + [43, 46]].reshape(2, -1, 2)

    def get_eye_rectangle_coordinates(self, out_shape):
        return self.get_eye_landmarks2d().mean(axis=1, dtype=int) - out_shape[::-1].reshape(1, 2) // 2

    def set_rotation(self, face_rotation_quaternion):
        self.rotation = quaternion_to_angle_axis(face_rotation_quaternion)
        return self

    def set_translation(self, key='nose'):
        self.translation = self.landmarks3D[key]
        return self

    def get_norm_vector_to_face(self):
        return cross(self.landmarks3D['chin'] - self.landmarks3D['eyes']['left']['center'],
                     self.landmarks3D['chin'] - self.landmarks3D['eyes']['right']['center'])
