import socketserver
import socket
import json
from app.estimation import GazeNet
from app.device.screen import Screen
from app.device.camera import Camera
from app.parser import ExperimentParser
from app.frame import Frame
from app.actor import Actor
from cv2 import waitKey
from app import *


dummy = {'FaceInfo': {'Landmarks'}}


class Handler_TCPServer(socketserver.BaseRequestHandler):
    """
    The TCP Server class for demonstration.

    Note: We need to implement the Handle method to exchange data
    with TCP client.

    """

    def handle(self):
        # self.request - TCP socket connected to the client
        self.data = self.request.recv(256).strip()
        json_data = self.data.split(b'\r\n\r\n')[-1]
        print(json_data)
        json_data = json.loads(json_data)
        print(json_data)
        # print(bytes(json_data['foo'], encoding='latin1'))
        # just send back data

        self.request.sendall(self.data+b'\r\n\r\n')


def run_server(host="localhost", port=5050):

    tcp_server = socketserver.TCPServer((host, port), Handler_TCPServer)
    tcp_server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        tcp_server.serve_forever()
    finally:
        tcp_server.server_close()


def ispressed(button, delay=1):
    return waitKey(delay) == button


def construct_scene_objects(origin_name, intrinsic_params, extrinsic_params):

    origin = Camera(name=origin_name, cam_dict=intrinsic_params['CAMERAS'].pop(origin_name))

    cams = {
        name: Camera(
            name=name,
            cam_dict=cam_data,
            extrinsic_matrix=(extrinsic_params[f'{name}_{origin.name}']),
            origin=origin
        ) for name, cam_data in intrinsic_params['CAMERAS'].items()
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
