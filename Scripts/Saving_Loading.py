import numpy as np
import csv
import tensorflow as tf
import sklearn.preprocessing


def save_learning_curve(eval_steps, returns, filename):
    steps_array = np.array(eval_steps)
    returns_array = np.array(returns)
    np.save(filename + '/eval_steps.npy', steps_array)
    np.save(filename + '/returns.npy', returns_array)

def save_clipfrac_divergence(clipfracs, kl_divergence, filename):
    clipfracs_array = np.array(clipfracs)
    kl_divergence_array = np.array(kl_divergence)
    np.save(filename + '/clipfracs.npy', clipfracs_array)
    np.save(filename + '/kl_divergence.npy', kl_divergence_array)


def load_learning_curve(filename):
    steps_array = np.load(filename + '/eval_steps.npy')
    returns_array = np.load(filename + '/returns.npy')
    eval_steps = list(steps_array)
    returns = list(returns_array)
    return eval_steps, returns


def save_scaler(scaler, filename):
    mean = scaler.mean_
    standard = np.sqrt(scaler.var_)
    np.save(filename + '/scale_state_mean.npy', mean)
    np.save(filename + '/scale_state_std.npy', standard)


def load_scaler(filename):
    mean = np.load(filename + '/scale_state_mean.npy')
    std = np.load(filename + '/scale_state_std.npy')
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = std
    scaler.var_ = std * std

    return scaler


def save_info(filename, input_type, input_dim, no_hidden, size_hidden, gamma, lr_critic, lr_actor, ep_length, dt,
              env_steps, no_episodes, V_swim, dir_target_x, dir_target_y, **kwargs):
    info = {'input': input_type, 'input_dim': input_dim, 'number_of_Hidden_Layers_Actor': no_hidden,
            'size_of_Hidden_Layers_Actor': size_hidden,
            'Gamma': gamma, 'lr_critic': lr_critic, 'lr_actor': lr_actor, 'Episode_Length': ep_length,
            'Delta_t': dt, 'Env_Steps': env_steps, 'Number_of_Episodes': no_episodes, 'V_Swim': V_swim,
            'Target_dir_x': dir_target_x, 'Target_dir_y': dir_target_y}

    info.update(kwargs)

    w = csv.writer(open(filename + "/Actor_Info.csv", "w"))
    for key, val in info.items():
        w.writerow([key, val])

def save_info_general(filename, info):
    w = csv.writer(open(filename + "/Actor_Info.csv", "w"))
    for key, val in info.items():
        w.writerow([key, val])

def save_parse(path, parsed_args):
    w = csv.writer(open(path + "/parsed_arguments.csv", "w"))
    for key, val in parsed_args.items():
        w.writerow([key, val])



def load_info(filename):
    with open(filename + "/Actor_Info.csv", mode='r') as inp:
        reader = csv.reader(inp)
        dict_from_csv = {rows[0]: rows[1] for rows in reader if bool(rows)}

    return dict_from_csv


def save_networks(filename, value_net, action_net):
    value_net.save(filename + '/Trained_Critic_Velocity')
    action_net.save_weights(filename + '/Trained_Actor_Weights')


def load_networks(filename, action_net):
    value_net = tf.keras.models.load_model(filename + '/Trained_Critic_Velocity')
    action_net.load_weights(filename + '/Trained_Actor_Weights')

    return value_net


def save_network_weights(filename, value_net, action_net):
    value_net.save_weights(filename + '/Trained_Critic_Weights')
    action_net.save_weights(filename + '/Trained_Actor_Weights')


def load_network_weights(filename, action_net, value_net):
    action_net.load_weights(filename + '/Trained_Actor_Weights')
    value_net.load_weights(filename + '/Trained_Critic_Weights')


def save_optimizers(filename, optimizer_actor, optimizer_critic):
    np.save(filename + '/optimizer_actor_weights.npy', optimizer_actor.get_weights())
    np.save(filename + '/optimizer_critic_weights.npy', optimizer_critic.get_weights())


def load_optimizers(filename, lr_actor, lr_critic, action_net, value_net):
    optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
    optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)

    with tf.name_scope(optimizer_actor._name):
        with tf.init_scope():
            optimizer_actor._create_all_weights(action_net.trainable_variables)

    with tf.name_scope(optimizer_critic._name):
        with tf.init_scope():
            optimizer_critic._create_all_weights(value_net.trainable_variables)

    optimizer_actor_weights = np.load(filename + '/optimizer_actor_weights.npy', allow_pickle=True)
    optimizer_critic_weights = np.load(filename + '/optimizer_critic_weights.npy', allow_pickle=True)

    optimizer_actor.set_weights(optimizer_actor_weights)
    optimizer_critic.set_weights(optimizer_critic_weights)

    return optimizer_actor, optimizer_critic
