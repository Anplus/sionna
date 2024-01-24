try:
    import sionna
except ImportError as e:
    import sionna
    print("import sionna")

import unittest
import numpy as np
import tensorflow as tf
import os
print(os.getcwd())
from sionna.rt import Scene, Transmitter, Receiver
from sionna.rt import load_scene
from sionna.rt import PlanarArray

# scene = load_scene(sionna.rt.scene.floor_wall)
# simple_reflector
scene = load_scene(sionna.rt.scene.simple_reflector)
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

scene.rx_array = scene.tx_array
tx_pos = [1, 0, 2]
rx_pos = [-1, 0, 2]
scene.add(Transmitter(name="tx",
                      position=tx_pos,
                      orientation=[0,0,0]))
scene.add(Receiver(name="rx",
                      position=rx_pos,
                      orientation=[0,0,0]))

paths = scene.compute_paths(los=True, reflection=True, scattering=False, diffraction=False, refraction=False, num_samples=100000)

print(paths.types)
paths.export("test_refraction_paths.obj")