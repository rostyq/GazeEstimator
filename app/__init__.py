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
from tqdm import tqdm

import socketserver
import numpy as np
import cv2


class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """
    freshChunkArrived = True
    payloadBuffer = []
    headerSize = 20
    payloadLength = 0

    def handle(self):
        # self.request is the TCP socket connected to the client

        e1 = cv2.getTickCount()
        data = self.request.recv(65536).strip()

        # receive image header first, then cat payload to the buffer
        header = data[0:self.headerSize]
        # first 4 bytes encode message type
        # next 8 bytes encode image width and height
        messageType = int.from_bytes(header[0:3], byteorder='little', signed=False)
        messageWidth = int.from_bytes(header[4:7], byteorder='little', signed=False)
        messageHeight = int.from_bytes(header[8:11], byteorder='little', signed=False)
        messageBpp = int.from_bytes(header[12:15], byteorder='little', signed=False)

        payloadBufferSize = messageWidth * messageHeight * messageBpp

        print(messageType)
        print(messageWidth)
        print(messageHeight)
        print(messageBpp)
        print(payloadBufferSize)
        print(payloadBufferSize, type(payloadBufferSize))
        payloadBuffer = bytearray(payloadBufferSize)
        print(len(payloadBuffer))
        numReceivedBytes = len(data) - self.headerSize
        payloadBuffer[0:numReceivedBytes] = data[self.headerSize:-1]

        while numReceivedBytes < payloadBufferSize:
            data = self.request.recv(65536)
            payloadBuffer[numReceivedBytes:numReceivedBytes + len(data)] = data
            numReceivedBytes += len(data)
            print("----")
            print(len(data))
            print(numReceivedBytes)
            print(len(payloadBuffer))

        print("Received image")
        print(len(payloadBuffer))
        #print("{} wrote:".format(self.client_address[0]))

        # display image

        image = np.frombuffer(payloadBuffer, np.uint8).reshape(messageHeight, messageWidth, messageBpp)

        e2 = cv2.getTickCount()

        print((e2 - e1) / cv2.getTickFrequency())
        cv2.imshow("image", image)
        cv2.waitKey(0)
        # not a first call
        print("data received")
        #print(self.data)
        # just send back the same data, but upper-cased
        self.request.sendall("Data received!".encode())

class Scene:
    def __init__(self, origin_name, intrinsic_params, extrinsic_params):
        params = {key: value for key, value in intrinsic_params['CAMERAS'].items() if key != origin_name}
        self.origin = Camera(name=origin_name, cam_dict=intrinsic_params['CAMERAS'][origin_name])

        self.cams = {
            name: Camera(
                name=name,
                cam_dict=cam_data,
                extrinsic_matrix=(extrinsic_params[f'{name}_{self.origin.name}']),
                origin=self.origin
            ) for name, cam_data in params.items()
        }
        self.cams[origin_name] = self.origin

        self.screens = {
            name: Screen(
                name=name,
                screen_dict=screen_data,
                extrinsic_matrix=extrinsic_params[f'{name}_{self.origin.name}'],
                origin=self.origin
            ) for name, screen_data in intrinsic_params['SCREENS'].items()
        }

    def to_dict(self):
        return {'screens':  {name: screen.to_dict() for name, screen in self.screens.items()},
                'cams': {name: cam.to_dict() for name, cam in self.cams.items()}}


def create_learning_dataset(save_path, parser, face_detector, scene, indices=None):
    save_path = Path.join(save_path, 'normalized_data', parser.session_code)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    learning_data = {'dataset': [], 'scene': scene.to_dict()}
    for (frames, data), index in parser.snapshots_iterate(indices=indices, progress_bar=True):
        if data['gazes']:
            frame_basler = frames['basler']
            actors_basler = face_detector.detect_actors(frame_basler, scene.origin)
            if len(actors_basler) == 0:
                continue
            actor_basler = actors_basler[0]
            actor_basler.set_landmarks3d_gazes(*data['gazes'], scene.screens['wall'])

            left_eye_frame, right_eye_frame = frame_basler.extract_eyes_from_actor(actor_basler,
                                                                                   resolution=(60, 36),
                                                                                   equalize_hist=True,
                                                                                   to_grayscale=True)

            cv2.imwrite(Path.join(save_path, f'{index}_left.png'), left_eye_frame)
            cv2.imwrite(Path.join(save_path, f'{index}_right.png'), right_eye_frame)

            learning_data['dataset'].append([actor_basler.to_learning_dataset(f'{index}_left.png',
                                                                              f'{index}_right.png',
                                                                              scene.cams['basler'])])

    with open(Path.join(save_path, 'normalized_dataset.json'), mode='w') as outfile:
        json.dump(learning_data, fp=outfile, indent=2)
    print(f"Dataset saved to {save_path}. Number of useful snapshots: {len(learning_data['dataset'])}")
