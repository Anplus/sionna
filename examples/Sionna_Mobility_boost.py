import sys
sys.path.append('..') # add sionna package path

try:
    import sionna
except ImportError as e:
    print('no sionna package found')

import time
import matplotlib.pyplot as plt
import scipy.io
import unittest
import numpy as np
import os
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

tf.random.set_seed(1) # Set global random seed for reproducibility
import matplotlib.pyplot as plt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera
from sionna.rt.utils import r_hat
from sionna.ofdm import ResourceGrid
from sionna.channel.utils import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.constants import SPEED_OF_LIGHT

if __name__ == "__main__":
    scene_folder = "/home/anplus/Documents/GitHub/diff-RT-channel-prediction/dataset/scene/princeton-human/"
    scene = 'princeton-processed-human.xml'
    scene_file = os.path.join(scene_folder, scene)
    scene = load_scene(scene_file)
    scene.add(Camera("cam", position=[7, 9, 2], look_at=[0, 0, 0]))
    human = scene.get("human")
    print("Human Position: ", human.position.numpy())
    print("Human Orientation: ", human.orientation.numpy())
    rx_pos = [1, 1, 1.3]
    tx_pos = [2.3, 5.9, 1.3]
    scene.add(Transmitter("tx", position=tx_pos, orientation=[0, 0, 0]))
    scene.add(Receiver(name="rx", position=rx_pos, orientation=[0, 0, 0]))

    scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, "dipole", "V")
    scene.rx_array = scene.tx_array
    NUM_SAMPLES_RAY = 1e6
    num_change = 1
    max_depth_settings = [2]
    time_consumptions = {depth: [] for depth in max_depth_settings}

    # mobility boost
    human.position = [1.52, 0.98136413, 0.8922081]
    depth = 3
    #_test = scene.compute_paths(max_depth=depth, num_samples=NUM_SAMPLES_RAY, diffraction=True, scattering=True, scat_keep_prob=0.001)
    traced_path_static = scene.trace_paths(max_depth=depth, num_samples=NUM_SAMPLES_RAY, reflection=False,diffraction=True, scattering=False, scat_keep_prob=0.001)
    _test = scene.compute_fields(*traced_path_static)
    # traced_paths = scene.trace_paths(max_depth=depth, num_samples=NUM_SAMPLES_RAY, diffraction=True, scattering=True, scat_keep_prob=0.001)
    human.position += [0, 0.1, 0]
    traced_paths_moving = scene.trace_paths_moving(max_depth=2, num_samples=600, moving_objects=human,
                                            diffraction=True, scattering=True, scat_keep_prob=0.001)
    paths_moving = scene.compute_fields(*traced_paths_moving)
    # Loop over each max_depth setting
    _test = scene.update_paths(*traced_path_static, _test, reflection=False, diffraction=True, scattering=False)
    paths_moving = paths_moving.merge_final(_test)
    for depth in max_depth_settings:
        human.position = [1.52, 0.98136413, 0.8922081]
        for i in range(num_change):
            start_time = time.time()
            human.position += [0, 0.01, 0]
            traced_paths = scene.trace_paths(max_depth=depth, num_samples=NUM_SAMPLES_RAY)
            paths = scene.compute_fields(*traced_paths)
            a, tau = paths.cir()
            elapsed_time = time.time() - start_time
            time_consumptions[depth].append(elapsed_time)

    time_consumptions_str_keys = {str(key): value for key, value in time_consumptions.items()}
    # scipy.io.savemat('time_consumptions.mat', time_consumptions_str_keys)
    img = scene.render(camera="cam", paths=None, show_paths=False, show_devices=False, num_samples=512)
    plt.show()
    plt.axis('off')  # Optional: Hide axis for better visualization