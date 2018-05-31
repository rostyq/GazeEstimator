from os import path as Path
import json
import os.path
from app.estimation import GazeNet
from app.device.screen import Screen
from app.device.camera import Camera
from app.parser import ExperimentParser
from app.estimation.actordetector import ActorDetector
from app.frame import Frame
from app.actor import Actor
from app import *
import numpy as np
import cv2


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

    def get_web_cam_image(frames, data):
        if data['gazes']:
            frame_basler = frames['basler']
            actors_basler = face_detector.detect_actors(frame_basler, scene.origin)
            if len(actors_basler) == 0:
                return None
            actor_basler = actors_basler[0]
            actor_basler.set_landmarks3d_gazes(*data['gazes'], scene.screens['wall'])
            gaze_line_basler = [
                actor_basler.landmarks3D['eyes']['left']['gaze'] + actor_basler.landmarks3D['eyes']['left']['center'],
                actor_basler.landmarks3D['eyes']['left']['center']]
            face_line_basler = [actor_basler.landmarks3D['nose'] + 100 * actor_basler.get_norm_vector_to_face(),
                                actor_basler.landmarks3D['nose']]

            frame = frames[cam_name]
            frame.project_lines(*gaze_line_basler)
            frame.project_lines(*face_line_basler, color=(0, 0, 255))
            frame.project_vectors(wall_points.reshape(-1, 3))
            return cv2.resize(frame.image, resolution)

    create_video(save_path, f'{cam_name}_{parser.session_code}.avi', resolution, 10.0, parser, get_web_cam_image, indices)


def create_wall_video(save_path, parser, face_detector, scene, indices=None):
    wall = scene.screens['wall']
    resolution = tuple((np.array([wall.resolution[1], wall.resolution[0]]) / 2).astype(int))

    def get_wall_image(frames, data):
        if data['gazes']:
            frame_basler = frames['basler']
            actors_basler = face_detector.detect_actors(frame_basler, scene.origin)
            if len(actors_basler) == 0:
                return None
            actor_basler = actors_basler[0]
            actor_basler.set_landmarks3d_gazes(*data['gazes'], scene.screens['wall'])
            gaze_line_basler = [
                actor_basler.landmarks3D['eyes']['left']['gaze'] + actor_basler.landmarks3D['eyes']['left']['center'],
                actor_basler.landmarks3D['eyes']['left']['center']]
            face_line_basler = [actor_basler.landmarks3D['nose'] + 100 * actor_basler.get_norm_vector_to_face(),
                                actor_basler.landmarks3D['nose']]

            gaze_intersection = wall.get_intersection_point_in_pixels(gaze_line_basler)
            face_intersection = wall.get_intersection_point_in_pixels(face_line_basler)
            image = wall.generate_image_with_circles(np.array([gaze_intersection, face_intersection]),
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