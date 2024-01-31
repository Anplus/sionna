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
from sionna.rt import Scene, Transmitter, Receiver, Camera
from sionna.rt import load_scene
from sionna.rt import PlanarArray

# scene = load_scene(sionna.rt.scene.floor_wall)
scene = load_scene('floor_wall_test.xml')
# simple_reflector
# scene = load_scene(sionna.rt.scene.simple_reflector)
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

scene.rx_array = scene.tx_array
tx_pos = [2, 0, 2]
rx_pos = [-2, 0, 2]
scene.add(Transmitter(name="tx",
                      position=tx_pos,
                      orientation=[0,0,0]))
scene.add(Receiver(name="rx",
                      position=rx_pos,
                      orientation=[0,0,0]))

paths = scene.compute_paths(los=True, reflection=False, scattering=True, diffraction=False, refraction=True, num_samples=1e5)

print(paths.vertices)
paths.export("test_refraction_paths.obj")


import matplotlib.pyplot as plt
# Create new camera with different configuration
resolution = [480, 320]
my_cam = Camera("my_cam", position=[1, 7, 2], look_at=[0, 0, 1])
scene.add(my_cam)
imgscene = scene.render("my_cam", paths=paths, resolution=resolution, num_samples=512)
plt.show()
plt.axis('off')  # Optional: Hide axis for better visualization
