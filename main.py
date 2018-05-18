from app.device import create_screen
from app.device import cam_from_dict
from config import *
import json
import socket
from app import *

with open('./extrinsic_params.json', 'r') as f:
    ext_params = json.load(f)

origin = cam_from_dict('ir', CAM_PARAMS.pop('ir'))

cams = [cam_from_dict(name=name, cam_dict=cam_data, relative=origin)
        for name, cam_data in CAM_PARAMS.items()]

for cam in cams:
    cam.set_extrinsic_from_matrix(ext_params[f'{cam.name}_{origin.name}'])

planes = [create_screen(name=plane,
                        extrinsic_matrix=ext_params[f'{plane}_{origin.name}'],
                        diagonal=None,
                        resolution=None,
                        relative=origin)
          for plane in ['wall', 'screen']]


if __name__ == "__main__":
    HOST, PORT = "localhost", 5055

    # Init the TCP server object, bind it to the localhost on 5055 port
    tcp_server = socketserver.TCPServer((HOST, PORT), Handler_TCPServer)
    # Activate the TCP server.
    tcp_server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        tcp_server.serve_forever()
    finally:
        tcp_server.server_close()
