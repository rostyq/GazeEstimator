from app.estimation import GazeNet
from app.device.screen import Screen
from app.device.camera import Camera
from app.parser import ExperimentParser
from app.frame import Frame
from app.actor import Actor
from app import *

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


def construct_scene_objects(origin_name, intrinsic_params, extrinsic_params):

    params = {key: value for key, value in intrinsic_params['CAMERAS'].items() if key != origin_name}
    origin = Camera(name=origin_name, cam_dict=intrinsic_params['CAMERAS'][origin_name])

    cams = {
        name: Camera(
            name=name,
            cam_dict=cam_data,
            extrinsic_matrix=(extrinsic_params[f'{name}_{origin.name}']),
            origin=origin
        ) for name, cam_data in params.items()
    }
    cams[origin_name] = origin

    screens = {
        name: Screen(
            name=name,
            screen_dict=screen_data,
            extrinsic_matrix=extrinsic_params[f'{name}_{origin.name}'],
            origin=origin
        ) for name, screen_data in intrinsic_params['SCREENS'].items()
    }

    return {'origin': origin, 'cameras': cams, 'screens': screens}
