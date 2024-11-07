from Scripts.Plotting import *
from Scripts.Models import *
from Scripts.Field import *
from Scripts.Gym_Envs import *
from Scripts.Saving_Loading import *
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, required=True, help="the directory to the model")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # This is an example of a script that plots trajectories in TGV of some learned policy. This script can be
    # modified with a few line changes in order to use all the visualization plots that are found in Plotting.py. For
    # 3D plots mayavi is used to visualize the trajectories.

    args = parse_args()

    file_name = args.file_name
    with open(file_name + '/parsed_arguments.csv', mode='r') as infile:
        reader = csv.reader(infile)
        info = {rows[0]: rows[1] for rows in reader}
    Uswim = float(info['uswim'])
    ep_length = int(info['N_steps'])
    time_step = float(info['time_step'])
    env_steps = int(info['env_steps'])
    dir_particle = json.loads(info['dir'])

    VelF_TGV = VelField(2 * np.pi, 2 * np.pi, 50, 50)
    VelF_TGV.Generate_TGV()

    ActivePart = ActiveParticle(1.0, 1.0, VelF_TGV, np.pi / 2, Uswim)

    total_time = env_steps * ep_length * time_step
    N_steps = int(total_time / 0.01)
    N_steps = int(N_steps * 2 / 3)

    env = GradEnv2D(Particle=ActivePart, Velocity_Field=VelF_TGV, Direction=dir_particle, time_step=0.01,
                    N_steps=N_steps, env_steps=1)

    input_dims = env.observation_space.shape[0]

    action_net = policy_network(input_dims, conversion=get_mean)
    value_net = value_function(input_dims)
    lazy_action_net = tf.function(action_net)
    lazy_value_net = tf.function(value_net)

    load_network_weights(file_name, action_net, value_net)

    scaler = load_scaler(file_name)

    fig, ax = plt.subplots(figsize=(0.75 * 6.4 / 2 + 1.0, 0.75 * 4.8))

    Plot_Learned_Traj_TGV(lazy_action_net, scaler, env, N_steps, fig, ax)
