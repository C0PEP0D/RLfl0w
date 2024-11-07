import numpy as np
import sklearn
import sklearn.preprocessing
import tensorflow as tf


def build_scaler(env, no_samples):
    state_space_samples = np.array(
        [env.reset(scaler=True) for x in range(no_samples)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)
    return scaler, state_space_samples

def build_scaler_turb(env, no_samples, func):
    rand_x = np.random.random(no_samples)*env.velf.x_range
    rand_y = np.random.random(no_samples)*env.velf.y_range
    rand_t = np.random.random(no_samples)*env.velf.t_range
    state_space_samples = np.array([func(rand_x[i], rand_y[i], rand_t[i]).flatten() for i in range(no_samples)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)
    return scaler


def load_velocity(n, emission_rate, lifetime, delta_timeit, max_time, dt):
    n_frames = int(max_time / (delta_timeit * dt))
    root_dir = './flow'
    file_dir = root_dir + "/emission-" + str(float(emission_rate)) + "_lifetime-" + str(
        float(lifetime)) + "_decay-continuous" + "_deltait-" + str(delta_timeit)
    u = np.zeros((n, n, n_frames))
    v = np.zeros((n, n, n_frames))
    for i in range(n_frames):
        loaded_array = np.load(file_dir + f'/timeit_{i * delta_timeit:010}_UV.npy')
        u[:, :, i] = loaded_array[0, :, :]
        v[:, :, i] = loaded_array[1, :, :]

    return u, v


def scale_state(state, scaler):
    if len(state.shape)==1:
        scaled = scaler.transform([state])
    else:
        scaled =  scaler.transform(state)

    return tf.convert_to_tensor(scaled)


def Evaluate_Policy(env, gamma, num_eps, scaler, *args, Random_Policy=False, Deterministic=False, three_D=False, env_steps=1,
                    **kwargs):
    episode_returns = []
    if (Random_Policy & Deterministic):
        raise ValueError('The policy cannot be both Random and Deterministic in the same call.')
    if (bool(args) & ('actor' in kwargs.keys())):
        raise ValueError('Cannot have both *args and an actor in the same call.')
    if (bool(args) & Random_Policy):
        raise ValueError('Cannot specify an action and a random policy in the same call.')
    if (('actor' in kwargs.keys()) & Random_Policy):
        raise ValueError('Cannot specify an actor and a random policy in the same call.')

    for episodes in range(num_eps):
        state = env.reset()
        ret = 0
        step = 0
        done = False
        while (not done):
            step += 1
            if args:
                if three_D:
                    action = args
                else:
                    action = args
                    action = np.squeeze(action, axis=0).reshape((1,))
            elif Deterministic:
                dist = kwargs['actor'](scale_state(state, scaler))
                if three_D:
                    act = dist.mean_direction
                    act = np.squeeze(act).reshape((3,))
                    theta = np.arccos(act[2])
                    phi = np.arctan2(act[1], act[0])
                    action = [theta / np.pi, phi / np.pi]
                else:
                    if env.type == 'Surf':
                        action = dist.loc
                        action = np.squeeze(action, axis=0).reshape((1,))
                    else:
                        action = dist.loc / np.pi
                        action = np.squeeze(action, axis=0).reshape((1,))
            elif Random_Policy:
                if three_D:
                    action = env.action_space.sample()
                else:
                    action = env.action_space.sample()
                    np.squeeze(action, axis=0).reshape((1,))
            else:
                if three_D:
                    act = kwargs['actor'](scale_state(state, scaler))
                    act = np.squeeze(act).reshape((3,))
                    theta = np.arccos(act[2])
                    phi = np.arctan2(act[1], act[0])
                    action = [theta / np.pi, phi / np.pi]
                else:
                    if env.type == 'Surf':
                        action = kwargs['actor'](scale_state(state, scaler))
                        action = np.squeeze(action, axis=0).reshape((1,))
                        # if action[0] <= 0.0:
                        #     action = np.array([0.0])
                        # if action[0] >= 10.0:
                        #     action = np.array([10.0])
                    else:
                        action = kwargs['actor'](scale_state(state, scaler)) / np.pi
                        action = np.squeeze(action, axis=0).reshape((1,))

            rew = 0
            for i in range(env_steps):
                next_state, reward, done, _ = env.step(action)
                rew += reward
                if done:
                    break
            state = next_state
            ret += rew * (gamma ** step)
        episode_returns.append(ret)
        print("Episode: {}".format(episodes))

    average_reward = sum(episode_returns) / num_eps

    return average_reward


def Get_Trajectory(env, scaler, *args, Random_Policy=False, Deterministic=False, three_D=False, env_steps=1, **kwargs):
    if (Random_Policy & Deterministic):
        raise ValueError('The policy cannot be both Random and Deterministic in the same call.')
    if (bool(args) & ('actor' in kwargs.keys())):
        raise ValueError('Cannot have both *args and an actor in the same call.')
    if (bool(args) & Random_Policy):
        raise ValueError('Cannot specify an action and a random policy in the same call.')
    if (('actor' in kwargs.keys()) & Random_Policy):
        raise ValueError('Cannot specify an actor and a random policy in the same call.')
    state = env.reset()
    if (('x' in kwargs.keys()) & ('y' in kwargs.keys())):
        env.particle.current_x = kwargs['x']
        env.particle.current_y = kwargs['y']
        env.particle.history_x = [kwargs['x']]
        env.particle.history_y = [kwargs['y']]
        if three_D:
            env.particle.current_z = kwargs['z']
            env.particle.history_z = [kwargs['z']]

    done = False
    while (not done):
        if args:
            if three_D:
                action = args
            else:
                action = args
                action = np.squeeze(action, axis=0).reshape((1,))
        elif Deterministic:
            dist = kwargs['actor'](scale_state(state, scaler))
            if three_D:
                act = dist.mean_direction
                act = np.squeeze(act).reshape((3,))
                theta = np.arccos(act[2])
                phi = np.arctan2(act[1], act[0])
                action = [theta / np.pi, phi / np.pi]
            else:
                action = dist.loc / np.pi
                action = np.squeeze(action, axis=0).reshape((1,))
        elif Random_Policy:
            if three_D:
                action = env.action_space.sample()
            else:
                action = env.action_space.sample()
                np.squeeze(action, axis=0).reshape((1,))
        else:
            if three_D:
                act = kwargs['actor'](scale_state(state, scaler))
                act = np.squeeze(act).reshape((3,))
                theta = np.arccos(act[2])
                phi = np.arctan2(act[1], act[0])
                action = [theta / np.pi, phi / np.pi]
            else:
                action = kwargs['actor'](scale_state(state, scaler)) / np.pi
                action = np.squeeze(action, axis=0).reshape((1,))

        for i in range(env_steps):
            next_state, _, done, _ = env.step(action)
            if done:
                break
        state = next_state
    if three_D:
        x, y, z = env.particle.get_absolute_positions()
        return x, y, z
    else:
        x, y = env.particle.get_absolute_positions()
        return x, y
