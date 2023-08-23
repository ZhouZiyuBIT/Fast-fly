import numpy as np
import yaml


class Gates():
    def __init__(self, yaml_f=""):
        
        self._pos = []
        self._rot = []
        self._N = 0
        self._shapes = []

        if yaml_f!="":
            self.load_from(yaml_f)

        for idx in range(self._N):
            self.add_shape(self._pos[idx], self._rot[idx])

    def add_gate(self, pos:list, rot=0):
        self._pos.append(pos)
        self._rot.append(rot)
        self.add_shape(pos, rot)
        self._N += 1

    def load_from(self, yaml_f):
        with open(yaml_f, 'r') as f:
            gf = yaml.load(f, Loader=yaml.FullLoader)
            self._pos = gf["pos"]
            self._rot = gf["rot"]
        self._N = len(self._pos)

    def add_shape(self, pos, rot):
        angles = np.linspace(0, 2*np.pi, 50)
        R = 0.35
        x = R*np.cos(angles)*np.cos(rot*np.pi/180)
        y = R*np.cos(angles)*np.sin(rot*np.pi/180)
        z = R*np.sin(angles)
        self._shapes.append([x+pos[0], y+pos[1], z+pos[2]])

