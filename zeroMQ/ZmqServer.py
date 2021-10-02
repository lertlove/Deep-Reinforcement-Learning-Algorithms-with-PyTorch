#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq
import json
import numpy as np
import threading

class ZmqServer(threading.Thread):

    def __init__(self, config):
        threading.Thread.__init__ (self)
        self.context = zmq.Context()
        self.environment = config.environment
        print("ZmqServer __init__")

    def run(self):
        print("ZmqServer run")
        socket = self.context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        
        while True:
            #  Wait for next request from client
            message = socket.recv_json()
            #  result = json.loads(message)

            print("Received command: %s" % message["command"])
            #  print("Received baseQP: %s" % message["baseQP"])
            func = getattr(self.environment, message["command"])
            reply = func(message)
            # listQP = message["listQP"]
            # meanQP = np.mean(message["listQP"])
            # print("Received listQP: %s" % listQP)
            # print("Received mean QP: %s" % meanQP)

            #  Do some 'work'
            time.sleep(1)
            
            # reply = f"We have got meanQP = {meanQP}"

            #  Send reply back to client
            socket.send(bytes(reply, 'utf-8'))

    # def start_experiment(self,message):
    #     print(f"start_experiment {message}")

    #     self.environment.start_experiment(message)

    #     reply = f"We have got start_experiment = {message}"
    #     return reply
    
    # def start_espisode(self,message):
    #     print(f"start_espisode {message}")
    #     reply = f"We have got start_espisode = {message}"
    #     return reply

    # def request_estimate_qp(self,message):
    #     print(f"request_estimate_qp {message}")
    #     reply = f"We have got request_estimate_qp = {message}"
    #     #return selected qp
    #     return reply

    # def after_apply_qp(self,message):
    #     print(f"after_apply_qp {message}")
    #     reply = f"We have got after_apply_qp = {message}"
    #     return reply

    # def end_episode(self,message):
    #     print(f"end_episode {message}")
    #     reply = f"We have got end_episode = {message}"
    #     return reply