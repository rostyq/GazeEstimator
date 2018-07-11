from os import path as Path
import json
import os.path
from app.estimation import GazeNet
from pygaze.display import Display
from app import *
import numpy as np
import cv2
import pypylon
import logging as log
import time
from numpy.linalg import norm


def create_video(save_path, name, resolution, frame_rate, parser, callback, indices=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(Path.join(save_path, name), fourcc, frame_rate, resolution)

    for (frames, data), index in parser.snapshots_iterate(indices=indices, progress_bar=True):
        image = callback(frames, data)
        if image is not None:
            out.write(image)
    out.release()
    print(f'Video is saved to {Path.join(save_path, name)}')


def create_gaze_video(save_path, parser, face_detector, scene, cam_name, indices=None):
    wall = scene.screens['wall']
    wall_points = np.mgrid[0:1.1:0.5, 0:1.1:0.5].reshape(2, -1).T
    wall_points = np.array([wall.point_to_origin(x, y) for (x, y) in wall_points])
    resolution = (640, 480)
    # model = GazeNet().init('checkpoints/model_700_0.0025.h5')

    def get_web_cam_image(frames, data):
        if data['face_points']:
            frame_basler = frames['basler']
            frame_kinect = frames['color']
            actors_basler = face_detector.detect_persons(frame_basler, scene.origin)
            actor_kinect = Person('kinect', origin=scene.origin)
            actor_kinect.set_kinect_landmarks3d(data['face_points'])
            if len(actors_basler) == 0:
                return None
            actor_basler = actors_basler[0]
            # actor_basler.set_landmarks3d_gazes(data['gazes'], scene.screens['wall'])

            # Estimation
            # left_gaze_line_basler = actor_basler.get_gaze_line(actor_basler.landmarks_3d['eyes']['left']['gaze'], key='left')
            # right_gaze_line_basler = actor_basler.get_gaze_line(actor_basler.landmarks_3d['eyes']['right']['gaze'], key='right')
            face_line_basler = [actor_basler.landmarks3D['nose'] + 100 * actor_basler.get_norm_vector_to_face(),
                                actor_basler.landmarks3D['nose']]
            face_line_color = [actor_kinect.landmarks_3d['nose'] + 100 * actor_kinect.get_norm_vector_to_face(),
                               actor_kinect.landmarks_3d['nose']]

            # left_gaze_intersection = scene.screens['wall'].get_intersection_point_origin(left_gaze_line_basler)
            # right_gaze_intersection = scene.screens['wall'].get_intersection_point_origin(right_gaze_line_basler)
            face_basler_intersection = scene.screens['wall'].get_intersection_point_origin(face_line_basler)
            face_color_intersection = scene.screens['wall'].get_intersection_point_origin(face_line_color)

            left_gaze_line_est = actor_kinect.get_gaze_line(data['est_gazes']['gazeLeft'], key='left')
            right_gaze_line_est = actor_kinect.get_gaze_line(data['est_gazes']['gazeRight'], key='right')
            left_est_intersection = wall.get_intersection_point_origin(left_gaze_line_est)
            right_est_intersection = wall.get_intersection_point_origin(right_gaze_line_est)
            # gaze_estimated_intersection = scene.screens['wall'].get_intersection_point_origin(gaze_line_estimated_basler)

            frame = frames[cam_name]
            # frame.project_lines(*face_line_basler, default_color=(255, 0, 255))
            frame.project_lines(*face_line_color, default_color=(0, 0, 255))
            frame.project_lines(*left_gaze_line_est, default_color=(255, 255, 255))
            frame.project_lines(*right_gaze_line_est, default_color=(255, 255, 255))
            frame.project_vectors(np.array([left_est_intersection, right_est_intersection]),
                                      default_color=(0, 255, 255))
            frame.project_vectors(wall_points.reshape(-1, 3))
            return cv2.resize(frame.image, resolution)

    create_video(save_path, f'{cam_name}_{parser.session_code}.avi', resolution, 10.0, parser, get_web_cam_image, indices)


def create_wall_video(save_path, parser, face_detector, scene, indices=None):
    wall = scene.screens['wall']
    resolution = tuple((np.array([wall.resolution[1], wall.resolution[0]]) / 2).astype(int))
    # model = GazeNet().init('checkpoints/model_700_0.0025.h5')

    def get_wall_image(frames, data):
        if data['gazes']:
            frame_basler = frames['basler']
            actors_basler = face_detector.detect_persons(frame_basler, scene.origin)
            if len(actors_basler) == 0:
                return None
            actor_basler = actors_basler[0]
            actor_basler.set_landmarks3d_gazes(data['gazes'], scene.screens['wall'])

            # Estimation
            left_eye = frame_basler.extract_eyes_from_actor(actor_basler, equalize_hist=True, to_grayscale=True)[0]
            gaze_line_basler = actor_basler.get_gaze_line(actor_basler.landmarks3D['eyes']['left']['gaze'], key='left')
            face_line_basler = [actor_basler.landmarks3D['nose'] + 100 * actor_basler.get_norm_vector_to_face(),
                                actor_basler.landmarks3D['nose']]
            # gaze_line_estimated_basler = actor_basler.get_gaze_line(
            #     frame_basler.camera.vectors_to_origin(
            #         model.estimate_gaze(left_eye, actor_basler.get_norm_vector_to_face())
            #     )
            # )

            gaze_intersection = wall.get_intersection_point_in_pixels(gaze_line_basler)
            face_intersection = wall.get_intersection_point_in_pixels(face_line_basler)
            # gaze_estimated_intersection = wall.get_intersection_point_in_pixels(gaze_line_estimated_basler)

            image = wall.generate_image_with_circles(np.array([gaze_intersection,
                                                               face_intersection,
                                                               ]),
                                                     padding=1000,
                                                     labels=['gaze', 'face_norm'],
                                                     colors=[(255, 0, 0), (0, 255, 0)])
            return cv2.resize(image, resolution)

    create_video(save_path, f'wall_{parser.session_code}.avi', resolution, 5.0, parser, get_wall_image, indices)


def validate_calibration(parser, scene, index, face_detector, cam_names = ['basler', 'color', 'web_cam']):

    def show_face_landmarks3d(frame, actor, title):
        img_copy = np.copy(frame.image)
        frame.project_vectors(actor.landmarks3D['chin'])
        frame.project_vectors(actor.landmarks3D['nose'])
        frame.project_vectors(actor.landmarks3D['eyes']['left']['center'])
        frame.project_vectors(actor.landmarks3D['eyes']['right']['center'])
        frame.project_vectors(actor.landmarks3D['eyes']['right']['rectangle'])
        frame.project_vectors(actor.landmarks3D['eyes']['left']['rectangle'])
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, frame.image)
        cv2.resizeWindow(title, 600, 360)
        frame.image = img_copy

    for (frames, data), index in parser.snapshots_iterate(indices=[index], progress_bar=False):
        frames_to_show = [frames[cam_name] for cam_name in cam_names]

        # Kinect landmarks
        if data['face_points'] is not None:
            actor_kinect = Person('kinect', origin=scene.origin)
            actor_kinect.set_kinect_landmarks3d(data['face_points'])
            print(f'Kinect landmarks: {actor_kinect.landmarks_3d}')
            for i, frame in enumerate(frames_to_show):
                show_face_landmarks3d(frame, actor_kinect, f'frame_{i}_actor_kinect')
        cv2.waitKey()

        # Basler lanmarks
        actors_basler = face_detector.detect_persons(frames['basler'], scene.origin)
        if len(actors_basler):
            actor_basler = actors_basler[0]
            print(f'Basler landmarks: {actor_basler.landmarks_3d}')
            for i, frame in enumerate(frames_to_show):
                show_face_landmarks3d(frame, actor_kinect, f'frame_{i}_actor_basler')
        cv2.waitKey()

        # Color landmarks
        face_detector.factor = 1
        actors_color = face_detector.detect_persons(frames['color'], scene.origin)
        if len(actors_color):
            actor_color= actors_color[0]
            print(f'Color landmarks: {actor_color.landmarks_3d}')
            for i, frame in enumerate(frames_to_show):
                show_face_landmarks3d(frame, actor_kinect, f'frame_{i}_actor_color')
        cv2.waitKey()

    cv2.destroyAllWindows()


def connect_gazepoint():
    from app.device.gaze_point import OpenGazeTrackerRETTNA
    tracker = OpenGazeTrackerRETTNA(None)
    tracker.start_recording()
    return tracker


def connect_basler(exposure_time=1000):
    print(pypylon.factory.find_devices())
    basler = pypylon.factory.create_device(pypylon.factory.find_devices()[0])
    basler.open()
    #print(basler.properties['ExposureTimeAbs'])
    basler.properties['ExposureTime'] = 80000
    # basler.properties['DeviceLinkThroughputLimitMode'] = 'Off'
    return basler


def show_point(point, scene):
    screen = scene.screens['wall']
    background = np.zeros((screen.resolution[1], screen.resolution[0], 3), dtype=np.uint8)
    point_in_pixels = screen.get_point_in_pixels(*point)
    Frame.draw_points(background, [point_in_pixels], colors=[(255, 255, 255)], radius=20)
    cv2.imshow("experiment", background)


def ispressed(button, delay=1):
    return cv2.waitKey(delay) == button

def init_experiment(save_path, session_code, size, scene, testing=False, path_to_model=None, screen='wall'):
    if not testing:
        # Save path
        save_path = Path.join(save_path, 'normalized_data' + size, session_code)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        else:
            raise Exception('Session already exists!')
    learning_data = {'dataset': [], 'scene': scene.to_dict()}

    # Model init
    # model = GazeNet().init('checkpoints/model_500_0.0039.h5')
    wall = scene.screens[screen]

    # Basler connection
    basler = connect_basler()

    # Window init
    cv2.namedWindow("experiment", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("experiment", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Gazepoint connection
    tracker = None
    if not testing:
        tracker = connect_gazepoint()

    model = None
    if testing:
        model = GazeNet().init(path_to_model)

    # Logging
    log.basicConfig(filename='log/experiment.log', level=log.INFO)

    return learning_data, wall, basler, tracker, model, save_path


def experiment_without_BRS(save_path, face_detector, scene, session_code, dataset_size=1000, size=''):

    learning_data, wall, basler, tracker, _, save_path = init_experiment(save_path, session_code, size, scene, screen='screen')
    index = 0
    lag = 1
    frames_basler = []
    gazes = []

    # os.spawnl(os.P_DETACH, 'mpv https://www.youtube.com/watch?v=ynHlGP6iSbI --fs --fs-screen=2')

    # Shooting
    try:
        while index < dataset_size:
            frame_basler = next(basler.grab_images(1))
            frame_time = time.time()
            sample = tracker.sample()
            gaze_time = time.time()
            log.info(f'GazePoint time: {gaze_time}, basler time: {frame_time}, difference: {frame_time - gaze_time}')
            if sample and frame_basler is not None and int(sample[-lag]['FPOGV']):
                print(f'Lag: {len(sample)} gazepoint samples. Frame {index}')
                gaze = {
                    'right': tuple(map(float, (sample[-lag]['LPOGX'], sample[-lag]['LPOGY']))),
                    'left': tuple(map(float, (sample[-lag]['RPOGX'], sample[-lag]['RPOGY'])))
                }
                frame_basler = Frame(scene.cams['basler'], cv2.flip(frame_basler, 1))
                # show_point(gaze, scene)
                # cv2.waitKey(1)
                frames_basler.append(frame_basler)
                gazes.append(gaze)
                index += 1
    finally:
        basler.close()
        tracker.stop_recording()
        tracker.close()
        cv2.destroyAllWindows()

    # Processing
    for index, (gaze, frame_basler) in tqdm(enumerate(zip(gazes, frames_basler))):
        actors_basler = face_detector.detect_persons(frame_basler, scene.origin)
        if len(actors_basler) == 0:
            print('No actors found!')
            continue
        actor_basler = actors_basler[0]
        actor_basler.set_landmarks3d_gazes(gaze, wall)

        left_eye_frame, right_eye_frame = frame_basler.extract_eyes_from_actor(actor_basler,
                                                                               resolution=(120, 72),
                                                                               equalize_hist=True,
                                                                               to_grayscale=False)
        # gaze_line_basler = actor_basler.get_gaze_line(actor_basler.landmarks_3d['eyes']['left']['gaze'], key='left')
        # gaze_intersection = wall.get_intersection_point_in_pixels(gaze_line_basler)

        # image = wall.generate_image_with_circles(np.array([gaze_intersection]),
        #                                          padding=0,
        #                                          labels=['gaze'],
        #                                          colors=[(255, 255, 255)])

        # cv2.imshow("experiment", image)
        # cv2.waitKey(1)

        cv2.imwrite(Path.join(save_path, f'{index}_left.png'), left_eye_frame)
        cv2.imwrite(Path.join(save_path, f'{index}_right.png'), right_eye_frame)

        learning_data['dataset'].append(
            actor_basler.to_learning_dataset(f'{index}_left.png', f'{index}_right.png', scene.cams['basler'])
        )

    # cv2.destroyAllWindows()

    with open(Path.join(save_path, 'normalized_dataset.json'), mode='w') as outfile:
        json.dump(learning_data, fp=outfile, indent=2)
    print(f"Dataset saved to {save_path}. Number of useful snapshots: {len(learning_data['dataset'])}")


def visualize_predict(face_detector, scene, path_to_model, back=None):

    _, wall, basler, tracker, model, _ = init_experiment(save_path=None, session_code=None, size='', scene=scene, testing=True,
                                                      path_to_model=path_to_model, screen='wall')
    try:
        while not ispressed(30):
            # sample = tracker.sample()
            frame_basler = next(basler.grab_images(1))
            if frame_basler is not None:

                # gaze = tuple(map(float, (sample[-1]['FPOGX'], sample[-1]['FPOGY'])))
                frame_basler = Frame(scene.cams['basler'], cv2.flip(frame_basler, 1)) #cv2.blur(cv2.flip(frame_basler, 1), (3, 3)))
                actors_basler = face_detector.detect_persons(frame_basler, scene.origin)
                if len(actors_basler) == 0:
                    print('No actors found!')
                    continue

                image = None

                for i, actor_basler in enumerate(actors_basler):

                    left_eye_frame, right_eye_frame = frame_basler.extract_eyes_from_actor(actor_basler,
                                                                                           resolution=(120, 72),
                                                                                           equalize_hist=True,
                                                                                           to_grayscale=False)
                    # gaze_line_basler = actor_basler.get_gaze_line(actor_basler.landmarks_3d['eyes']['left']['gaze'])
                    # gaze_intersection = wall.get_intersection_point_in_pixels(gaze_line_basler)
                    norm_to_face = np.linalg.inv(frame_basler.camera.get_rotation_matrix()) @ (actor_basler.get_norm_vector_to_face() / norm(actor_basler.get_norm_vector_to_face())).reshape(3, -1)
                    gaze_line_left_estimated_basler = actor_basler.get_gaze_line(
                        frame_basler.camera.get_rotation_matrix() @ model.estimate_gaze(left_eye_frame, norm_to_face).reshape(3, -1),
                        key='left'
                    )
                    gaze_line_right_estimated_basler = actor_basler.get_gaze_line(
                        frame_basler.camera.get_rotation_matrix() @
                        (model.estimate_gaze(cv2.flip(right_eye_frame, 1), norm_to_face * np.array([[-1], [1], [1]])) * np.array([-1, 1, 1])).reshape(3, -1),
                        key='right'
                    )

                    face_line_basler = [actor_basler.landmarks3D['nose'] + 50 * actor_basler.get_norm_vector_to_face(),
                                        actor_basler.landmarks3D['nose']]
                    gaze_left_estimated_intersection = wall.get_intersection_point_in_pixels(gaze_line_left_estimated_basler)
                    gaze_right_estimated_intersection = wall.get_intersection_point_in_pixels(gaze_line_right_estimated_basler)
                    gaze_estimated_intersection = np.array([gaze_left_estimated_intersection, gaze_right_estimated_intersection]).mean(axis=0)
                    face_intersection = wall.get_intersection_point_in_pixels(face_line_basler)
                    image = wall.generate_image_with_circles(np.array([face_intersection, gaze_estimated_intersection,#]),
                                                                       gaze_left_estimated_intersection,
                                                                       gaze_right_estimated_intersection]),
                                                             padding=0,
                                                             labels=[f'FN{i}', f'GA{i}',
                                                                     f'GL{i}', f'GR{i}'],
                                                             colors=[(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)],
                                                             image=back)
                    image[:72, :120], image[:72, 120:240] = cv2.cvtColor(right_eye_frame, cv2.COLOR_GRAY2BGR), \
                                                            cv2.cvtColor(left_eye_frame, cv2.COLOR_GRAY2BGR)

                cv2.imshow("experiment", image)
                # cv2.imshow("experiment", cv2.cvtColor(frame_basler.image, cv2.COLOR_GRAY2BGR) + cv2.resize(image, (1296, 972)))
    finally:
        basler.close()
        # tracker.stop_recording()
        cv2.destroyAllWindows()


