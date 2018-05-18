import socketserver
from app.estimation import GazeNet
from numpy import zeros
import json
from json.decoder import JSONDecodeError

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
