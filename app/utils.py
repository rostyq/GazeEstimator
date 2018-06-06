from os import path as Path
import json
from app.device.gaze_point import OpenGazeTrackerRETTNA
import os.path
from pygaze.display import Display
from app import *
import numpy as np
import cv2
import pypylon
import time

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
    model = GazeNet().init('checkpoints/model_700_0.0025.h5')

    def get_web_cam_image(frames, data):
        if data['gazes']:
            frame_basler = frames['basler']
            actors_basler = face_detector.detect_actors(frame_basler, scene.origin)
            if len(actors_basler) == 0:
                return None
            actor_basler = actors_basler[0]
            actor_basler.set_landmarks3d_gazes(*data['gazes'], scene.screens['wall'])

            # Estimation
            left_eye = frame_basler.extract_eyes_from_actor(actor_basler, equalize_hist=True, to_grayscale=True)[0]
            gaze_line_basler = [
                actor_basler.landmarks3D['eyes']['left']['gaze'] + actor_basler.landmarks3D['eyes']['left']['center'],
                actor_basler.landmarks3D['eyes']['left']['center']]
            gaze_line_estimated_basler = [
                model.estimate_gaze(left_eye, actor_basler.get_norm_vector_to_face()) + actor_basler.landmarks3D['eyes']['left']['center'],
                actor_basler.landmarks3D['eyes']['left']['center']
            ]
            face_line_basler = [actor_basler.landmarks3D['nose'] + 100 * actor_basler.get_norm_vector_to_face(),
                                actor_basler.landmarks3D['nose']]
            gaze_intersection = scene.screens['wall'].get_intersection_point_origin(gaze_line_basler)
            face_intersection = scene.screens['wall'].get_intersection_point_origin(face_line_basler)
            gaze_estimated_intersection = scene.screens['wall'].get_intersection_point_origin(gaze_line_estimated_basler)

            frame = frames[cam_name]
            frame.project_vectors(gaze_intersection)
            frame.project_vectors(face_intersection, default_color=(0, 0, 255))
            frame.project_vectors(gaze_estimated_intersection, default_color=(0, 255, 0))
            frame.project_lines(*gaze_line_basler, default_color=(0, 255, 0))
            frame.project_lines(*face_line_basler, default_color=(0, 0, 255))
            frame.project_vectors(wall_points.reshape(-1, 3))
            return cv2.resize(frame.image, resolution)

    create_video(save_path, f'{cam_name}_{parser.session_code}.avi', resolution, 10.0, parser, get_web_cam_image, indices)


def create_wall_video(save_path, parser, face_detector, scene, indices=None):
    wall = scene.screens['wall']
    resolution = tuple((np.array([wall.resolution[1], wall.resolution[0]]) / 2).astype(int))
    model = GazeNet().init('checkpoints/model_700_0.0025.h5')

    def get_wall_image(frames, data):
        if data['gazes']:
            frame_basler = frames['basler']
            actors_basler = face_detector.detect_actors(frame_basler, scene.origin)
            if len(actors_basler) == 0:
                return None
            actor_basler = actors_basler[0]
            actor_basler.set_landmarks3d_gazes(*data['gazes'], scene.screens['wall'])

            # Estimation
            left_eye = frame_basler.extract_eyes_from_actor(actor_basler, equalize_hist=True, to_grayscale=True)[0]
            gaze_line_basler = [
                actor_basler.landmarks3D['eyes']['left']['gaze'] + actor_basler.landmarks3D['eyes']['left']['center'],
                actor_basler.landmarks3D['eyes']['left']['center']]
            face_line_basler = [actor_basler.landmarks3D['nose'] + 100 * actor_basler.get_norm_vector_to_face(),
                                actor_basler.landmarks3D['nose']]
            gaze_line_estimated_basler = [
                model.estimate_gaze(left_eye, actor_basler.get_norm_vector_to_face()) +
                actor_basler.landmarks3D['eyes']['left']['center'],
                actor_basler.landmarks3D['eyes']['left']['center']
            ]

            gaze_intersection = wall.get_intersection_point_in_pixels(gaze_line_basler)
            face_intersection = wall.get_intersection_point_in_pixels(face_line_basler)
            gaze_estimated_intersection = wall.get_intersection_point_in_pixels(gaze_line_estimated_basler)

            image = wall.generate_image_with_circles(np.array([gaze_intersection,
                                                               face_intersection,
                                                               gaze_estimated_intersection]),
                                                     padding=1000,
                                                     labels=['gaze', 'face_norm', 'estimated_left_gaze'],
                                                     colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)])
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
            actor_kinect = Actor('kinect', origin=scene.origin)
            actor_kinect.set_landmarks3d(data['face_points'])
            print(f'Kinect landmarks: {actor_kinect.landmarks3D}')
            for i, frame in enumerate(frames_to_show):
                show_face_landmarks3d(frame, actor_kinect, f'frame_{i}_actor_kinect')
        cv2.waitKey()

        # Basler lanmarks
        actors_basler = face_detector.detect_actors(frames['basler'], scene.origin)
        if len(actors_basler):
            actor_basler = actors_basler[0]
            print(f'Basler landmarks: {actor_basler.landmarks3D}')
            for i, frame in enumerate(frames_to_show):
                show_face_landmarks3d(frame, actor_kinect, f'frame_{i}_actor_basler')
        cv2.waitKey()

        # Color landmarks
        face_detector.factor = 1
        actors_color = face_detector.detect_actors(frames['color'], scene.origin)
        if len(actors_color):
            actor_color= actors_color[0]
            print(f'Color landmarks: {actor_color.landmarks3D}')
            for i, frame in enumerate(frames_to_show):
                show_face_landmarks3d(frame, actor_kinect, f'frame_{i}_actor_color')
        cv2.waitKey()

    cv2.destroyAllWindows()


def connect_gazepoint():
    disp = Display()
    tracker = OpenGazeTrackerRETTNA(disp)
    tracker.start_recording()
    disp.close()
    return tracker


def connect_basler():
    print(pypylon.factory.find_devices())
    basler = pypylon.factory.create_device(pypylon.factory.find_devices()[0])
    basler.open()
    return basler


def show_point(point, scene):
    screen = scene.screens['wall']
    background = np.zeros((screen.resolution[1], screen.resolution[0], 3), dtype=np.uint8)
    point_in_pixels = screen.get_point_in_pixels(*point)
    Frame.draw_points(background, [point_in_pixels], colors=[(255, 255, 255)], radius=20)
    cv2.imshow("experiment", background)

def ispressed(button, delay=1):
    return cv2.waitKey(delay) == button


def experiment_without_BRS(save_path, face_detector, scene, session_code, predict=False):

    # Save path
    save_path = Path.join(save_path, 'normalized_data', session_code)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    learning_data = {'dataset': [], 'scene': scene.to_dict()}

    # Model init
    model = GazeNet().init('checkpoints/model_500_0.0039.h5')
    wall = scene.screens['wall']

    # Basler connection
    basler = connect_basler()

    # Window init
    cv2.namedWindow("experiment", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("experiment", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    index = 0

    # Gazepoint connection
    tracker = connect_gazepoint()

    try:
        while not ispressed(27):

            sample = tracker.sample()
            frame_basler = next(basler.grab_images(1))
            start = time.time()
            if sample and frame_basler is not None and int(sample[-1]['FPOGV']):
                print(len(sample))

                gaze = tuple(map(float, (sample[-1]['FPOGX'], sample[-1]['FPOGY'])))
                frame_basler = Frame(scene.cams['basler'], frame_basler)
                actors_basler = face_detector.detect_actors(frame_basler, scene.origin)
                if len(actors_basler) == 0:
                    print('No actors found!')
                    continue
                actor_basler = actors_basler[0]
                actor_basler.set_landmarks3d_gazes(*gaze, scene.screens['wall'])

                left_eye_frame, right_eye_frame = frame_basler.extract_eyes_from_actor(actor_basler,
                                                                                       resolution=(60, 36),
                                                                                       equalize_hist=True,
                                                                                       to_grayscale=False)
                gaze_line_basler = [
                    actor_basler.landmarks3D['eyes']['left']['gaze'] + actor_basler.landmarks3D['eyes']['left'][
                        'center'],
                    actor_basler.landmarks3D['eyes']['left']['center']]
                gaze_intersection = wall.get_intersection_point_in_pixels(gaze_line_basler)

                if predict:
                    gaze_line_estimated_basler = [
                        frame_basler.camera.vectors_to_origin(
                            model.estimate_gaze(left_eye_frame, actor_basler.get_norm_vector_to_face())).reshape(3) +
                        actor_basler.landmarks3D['eyes']['left']['center'],
                        actor_basler.landmarks3D['eyes']['left']['center']
                    ]

                    gaze_estimated_intersection = wall.get_intersection_point_in_pixels(gaze_line_estimated_basler)
                    image = wall.generate_image_with_circles(np.array([gaze_intersection, gaze_estimated_intersection]),
                                                             padding=0,
                                                             labels=['gaze', 'estimated_left_gaze'],
                                                             colors=[(255, 0, 0), (0, 255, 0)])
                else:
                    image = wall.generate_image_with_circles(np.array([gaze_intersection]),
                                                             padding=0,
                                                             labels=['gaze'],
                                                             colors=[(255, 255, 255)])

                cv2.imshow("experiment", image)

                if not predict:
                    cv2.imwrite(Path.join(save_path, f'{index}_left.png'), left_eye_frame)
                    cv2.imwrite(Path.join(save_path, f'{index}_right.png'), right_eye_frame)

                    learning_data['dataset'].append(
                        [actor_basler.to_learning_dataset(f'{index}_left.png',
                                                          f'{index}_right.png',
                                                          scene.cams['basler'])])
                index += 1
                print(time.time() - start)

    finally:
        basler.close()
        tracker.stop_recording()
        cv2.destroyAllWindows()

    with open(Path.join(save_path, 'normalized_dataset.json'), mode='w') as outfile:
        json.dump(learning_data, fp=outfile, indent=2)
    print(f"Dataset saved to {save_path}. Number of useful snapshots: {len(learning_data['dataset'])}")



