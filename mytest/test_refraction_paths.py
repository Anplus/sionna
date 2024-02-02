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

paths = scene.compute_paths(los=False, reflection=True, scattering=True, diffraction=False, refraction=True, num_samples=1e5)

print(paths.vertices)
paths.export("test_refraction_paths.obj")
a, tau = paths.cir()
print(a.shape)
# a = tf.squeeze(a)
# tau = tf.squeeze(tau)
# a_ = a.numpy()
# tau_ = tau.numpy()
# tau_, a_ = zip(*sorted(zip(tau_, a_)))
# index = 3
# tau = np.array(tau_)[index:]
# a = np.array(a_)[index:]
print(a, tau)
import matplotlib.pyplot as plt
# Create new camera with different configuration
resolution = [480, 320]
my_cam = Camera("my_cam", position=[1, 7, 2], look_at=[0, 0, 1])
scene.add(my_cam)
imgscene = scene.render("my_cam", paths=paths, resolution=resolution, num_samples=512)
plt.show()
plt.axis('off')  # Optional: Hide axis for better visualization


t = tau[0,0,:]/1e-9 # Scale to ns
a_abs = np.abs(a)[0,0,0,0,0,:,0]
print(a_abs.shape, t.shape)
a_max = np.max(a_abs)

# And plot the CIR
plt.figure()
plt.title("Channel impulse response realization")

plt.stem(t, a_abs)
plt.xlim([0, np.max(t)])
# plt.ylim([-2e-6, a_max*1.1])
plt.xlabel(r"$\tau$ [ns]")
plt.ylabel(r"$|a|$")
plt.show()
