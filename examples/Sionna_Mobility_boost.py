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

def plt_cir(tau, h, label, linefmt, markerfmt, basefmt):
    mask = tau != -1
    h = h[mask]
    tau = tau[mask]
    tau = np.concatenate([(0.,), tau, (np.max(tau) * 1.1,)])
    h = np.concatenate([(np.nan,), h, (np.nan,)])
    plt.stem(tau * 1e9, np.abs(h), linefmt=linefmt, markerfmt=markerfmt, basefmt=basefmt, label=label)
    plt.xlim(0, 80)


if __name__ == "__main__":
    scene_folder = "/home/anplus/Documents/GitHub/diff-RT-channel-prediction/dataset/scene/princeton-human/"
    scene = 'princeton-processed-human.xml'
    scene_file = os.path.join(scene_folder, scene)
    scene = load_scene(scene_file)
    scene.add(Camera("cam", position=[7, 9, 2], look_at=[0, 0, 0]))
    human = scene.get("human")
    print("Human Position: ", human.position.numpy())
    print("Human Orientation: ", human.orientation.numpy())

    rx_pos = np.array([2, 1, 1.5])
    tx_pos = np.array([2, 6, 1.5])
    scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, "dipole", "V")
    scene.rx_array = scene.tx_array
    scene.add(Transmitter("tx", position=tx_pos, orientation=[0, 0, 0]))
    scene.add(Receiver(name="rx", position=rx_pos, orientation=[0, 0, 0]))

    NUM_SAMPLES_RAY = 2e6
    num_change = 1
    max_depth_settings = [2]
    time_consumptions = {depth: [] for depth in max_depth_settings}

    # START static
    human.position = [1, 1, 0.9] #[1.52, 0.98136413, 0.8922081]
    depth = 3
    reflection_flag = True
    diffraction_flag = True
    scattering_flag = False
    #_test = scene.compute_paths(max_depth=depth, num_samples=NUM_SAMPLES_RAY, diffraction=True, scattering=True, scat_keep_prob=0.001)
    traced_path_static = scene.trace_paths(max_depth=depth, num_samples=NUM_SAMPLES_RAY, \
                                        reflection=reflection_flag,diffraction=diffraction_flag, scattering=scattering_flag, scat_keep_prob=0.001)
    cir_static = scene.compute_fields(*traced_path_static)
    cir_static.normalize_delays = False
    a_nonblock, tau_nonblock = cir_static.cir()
    fig = plt.figure()
    plt_cir(np.squeeze(tau_nonblock), np.squeeze(a_nonblock), label='Static', linefmt='C1-', markerfmt='C1o',
            basefmt='C1-')
    plt.legend()
    plt.xlabel(r"Delay $\tau$ (ns)")
    plt.ylabel(r"$|h|^2$")
    plt.title("Comparison of Channel Impulse Responses")
    plt.show()

    cir_static.normalize_delays = True

    # NEW POSITION
    # traced_paths = scene.trace_paths(max_depth=depth, num_samples=NUM_SAMPLES_RAY, diffraction=True, scattering=True, scat_keep_prob=0.001)
    human.position = [2, 2.9, 1]
    ##############
    img = scene.render(camera="cam", paths=None, show_paths=False, show_devices=True, num_samples=512)
    plt.show()
    plt.axis('off')  # Optional: Hide axis for better visualization
    ##############
    paths_moving_naive = scene.compute_paths(max_depth=depth, num_samples=NUM_SAMPLES_RAY, \
                                        reflection=reflection_flag,diffraction=diffraction_flag, scattering=scattering_flag, scat_keep_prob=0.001)
    traced_paths_moving = scene.trace_paths_moving(max_depth=2, num_samples=1e5, moving_objects=human, reflection=reflection_flag,
                                            diffraction=diffraction_flag, scattering=scattering_flag, scat_keep_prob=0.001)
    paths_moving = scene.compute_fields(*traced_paths_moving)
    # Loop over each max_depth setting
    cir_static_ = scene.update_paths(*traced_path_static, cir_static, reflection=reflection_flag,
                                     diffraction=diffraction_flag, scattering=scattering_flag)
    #paths_moving.normalize_delays = False
    paths_moving = paths_moving.merge_final(cir_static_)
    paths_moving.normalize_delays = False
    paths_moving_naive.normalize_delays = False
    a_boost_s, tau_boost_s = cir_static_.cir()
    a_boost, tau_boost = paths_moving.cir()
    a_naive, tau_naive = paths_moving_naive.cir()
    fig = plt.figure()
    plt_cir(np.squeeze(tau_naive), np.squeeze(a_naive), label='Naive', linefmt='C1-', markerfmt='C1o',
            basefmt='C1-')
    plt.legend()
    plt.xlabel(r"Delay $\tau$ (ns)")
    plt.ylabel(r"$|h|^2$")
    plt.title("Comparison of Channel Impulse Responses")
    plt.show()
    #####################################
    plt.figure()
    plt_cir(np.squeeze(tau_boost), np.squeeze(a_boost), label='Boost', linefmt='C0-', markerfmt='C0o',
            basefmt='C0-')
    plt.legend()
    plt.xlabel(r"Delay $\tau$ (ns)")
    plt.ylabel(r"$|h|^2$")
    plt.title("Comparison of Channel Impulse Responses")
    plt.show()
    ######################################
    for depth in max_depth_settings:
        human.position = [0.5, 1, 0.9]
        for i in range(num_change):
            start_time = time.time()
            human.position += [0, 0.5, 0]
            traced_paths = scene.trace_paths(max_depth=depth, num_samples=NUM_SAMPLES_RAY)
            paths = scene.compute_fields(*traced_paths)
            a, tau = paths.cir()
            elapsed_time = time.time() - start_time
            time_consumptions[depth].append(elapsed_time)

    time_consumptions_str_keys = {str(key): value for key, value in time_consumptions.items()}
    # scipy.io.savemat('time_consumptions.mat', time_consumptions_str_keys)
