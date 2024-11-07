import json
from Scripts.Useful import *
from Scripts.Models import *
from Scripts.Field import *
from Scripts.Gym_Envs import *
from Scripts.Saving_Loading import *
from Scripts.Plotting import *
from stable_baselines3.common.env_util import make_vec_env
from bisect import bisect
import argparse


def get_discrete_state_custom(state, intervals):
    disc_state = np.zeros(state.shape)
    for ind, obs in enumerate(state):
        disc_state[ind] = bisect(intervals[ind], obs)

    disc_state = int(sum(disc_state * np.array([len(intervals[0]) + 1] * len(intervals)) ** np.arange(len(intervals))))
    return disc_state


def get_env_action2D(disc_action):
    theta = None
    if disc_action == 0:
        theta = 0.0
    elif disc_action == 1:
        theta = np.pi / 2
    elif disc_action == 2:
        theta = np.pi
    elif disc_action == 3:
        theta = 3 * np.pi / 2

    return [theta]


def get_env_action3D(disc_action):
    direction = None
    if disc_action == 0:
        direction = [1.0, 0.0, 0.0]
    elif disc_action == 1:
        direction = [0.0, 0.0, 1.0]
    elif disc_action == 2:
        direction = [-1.0, 0.0, 0.0]
    elif disc_action == 3:
        direction = [0.0, 0.0, -1.0]
    elif disc_action == 4:
        direction = [0.0, 1.0, 0.0]
    elif disc_action == 5:
        direction = [0.0, -1.0, 0.0]

    return direction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, required=True, help="the directory to the model")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
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

    total_steps = ep_length * env_steps

    if info['TGV']=='True':
        VelF = VelField(2 * np.pi, 2 * np.pi, 50, 50)
        VelF.Generate_TGV()
        ActivePart = ActiveParticle(1.0, 1.0, VelF, np.pi / 2, Uswim)
        if info['vort']=='True':
            env = VortEnv2D(Particle=ActivePart, Velocity_Field=VelF, Direction=dir_particle, time_step=time_step,
                            N_steps=total_steps, env_steps=1)
            envs = make_vec_env(VortEnv2D, n_envs=1000,
                                env_kwargs={"Velocity_Field": VelF, "Particle": ActivePart, "Direction": dir_particle,
                                            "time_step": time_step,
                                            "N_steps": total_steps, "env_steps": 1})
        elif info['grad']=='True':
            env = GradEnv2D(Particle=ActivePart, Velocity_Field=VelF, Direction=dir_particle, time_step=time_step,
                            N_steps=total_steps, env_steps=1)
            envs = make_vec_env(GradEnv2D, n_envs=1000,
                                env_kwargs={"Velocity_Field": VelF, "Particle": ActivePart, "Direction": dir_particle,
                                            "time_step": time_step,
                                            "N_steps": total_steps, "env_steps": 1})
        else:
            raise RuntimeError("Did not specify an environment")

    elif info['ABC']=='True':
        VelF = VelField3D(2 * np.pi, 2 * np.pi, 2 * np.pi, 50, 50, 50)
        three_D = True
        A = np.sqrt(3)
        B = np.sqrt(2)
        C = np.float64(1.0)
        VelF.Generate_ABC(A, B, C)
        ActivePart = ActiveParticle3D(1.0, 1.0, 1.0, VelF, np.pi / 4, np.pi / 4, Uswim)
        if info['vort']=='True':
            env = VortEnv3D(Particle=ActivePart, Velocity_Field=VelF, Direction=dir_particle, time_step=time_step,
                            N_steps=total_steps, env_steps=1)
            envs = make_vec_env(VortEnv3D, n_envs=1000,
                                env_kwargs={"Velocity_Field": VelF, "Particle": ActivePart, "Direction": dir_particle,
                                            "time_step": time_step,
                                            "N_steps": total_steps, "env_steps": 1})
        else:
            raise RuntimeError("Did not specify an environment")

    elif  info['turb']=='True':
        n = 256
        EMISSION_RATE = 1.0
        LIFETIME = 5.0
        DELTA_TIMEIT = 50
        MAX_TIME = 1 * DELTA_TIMEIT
        DT = 1e-3
        N_FRAMES = int(MAX_TIME / (DELTA_TIMEIT * DT))
        u, v = load_velocity(n, EMISSION_RATE, LIFETIME, DELTA_TIMEIT, MAX_TIME, DT)
        VelF = VelFieldTran(2 * np.pi, 2 * np.pi, MAX_TIME, n, n, N_FRAMES, u, v)
        ActivePart = ActiveParticleTran(5.1, 2.5, 0.0, VelF, np.pi / 2, Uswim, pedley=float(info['pedley']))
        if info['grad']=='True':
            env = GradEnvTran(Particle=ActivePart, Velocity_Field=VelF, Direction=dir_particle, time_step=time_step,
                            N_steps=total_steps, env_steps=1)
            envs = make_vec_env(GradEnvTran, n_envs=1000,
                                env_kwargs={"Velocity_Field": VelF, "Particle": ActivePart, "Direction": dir_particle,
                                            "time_step": time_step,
                                            "N_steps": total_steps, "env_steps": 1})

        elif info['vel']=='True':
            env = VelEnvTran(Particle=ActivePart, Velocity_Field=VelF, Direction=dir_particle, time_step=args.time_step,
                             N_steps=args.N_steps, env_steps=args.env_steps)
            envs = make_vec_env(VelEnvTran, n_envs=1000,
                                env_kwargs={"Velocity_Field": VelF, "Particle": ActivePart, "Direction": dir_particle,
                                            "time_step": time_step,
                                            "N_steps": total_steps, "env_steps": 1})

        else:
            raise RuntimeError("Did not specify an environment")

    else:
        raise RuntimeError("Did not specify a flow to learn on!")





    if  info['A2C']=='True' or info['PPO']=='True':
        input_dims = env.observation_space.shape[0]
        if info['lstm']=='True':
            action_net = policy_network_lstm(input_dims, 1000, conversion=get_mean)
            value_net = value_function_lstm(input_dims, 1000)
        else:
            if info['ABC']=='True':
                action_net = policy_network3D(input_dims, conversion=get_mean3D)
                value_net = value_function(input_dims)
            else:
                action_net = policy_network(input_dims, conversion=get_mean)
                value_net = value_function(input_dims)

        lazy_action_net = tf.function(action_net)
        lazy_value_net = tf.function(value_net)

        load_network_weights(file_name, action_net, value_net)
        scaler = load_scaler(file_name)

        returns = []
        episodes = 0
        a = 0.1
        prob = 100.0
        next_state = np.zeros((len(envs.envs), envs.envs[0].observation_space.shape[0]))
        _ = envs.reset()
        for i in range(next_state.shape[0]):
            start_time = np.random.rand() * (env.velf.t_range * 1 / 4 - 5.1) + env.velf.t_range * 3 / 4
            envs.envs[i].particle.current_t = start_time
            next_state[i, :] = env.pos_to_state(envs.envs[i].particle.current_x, envs.envs[i].particle.current_y,
                                                envs.envs[i].particle.current_t)

        # next_state = scale_state(envs.reset(), scaler)
        next_state = scale_state(next_state, scaler)
        S1 = 0.0
        S2 = 0.0
        if info['lstm']=='True':
            action_net.reset_states()

        counter = 0
        while prob >= 0.05:
            state = next_state
            if info['lstm']=='True':
                actions = tf.squeeze(lazy_action_net(tf.expand_dims(state, axis=1)), axis=1)
            else:
                actions = lazy_action_net(state)
            next_state, reward, next_done, infos = envs.step(actions.numpy())
            #     next_state = scale_state(np.concatenate((next_state,-next_state[:,:1]), axis=1), scaler)
            if any(next_done):
                for ind, elem in enumerate(next_done):
                    if elem == True:
                        if info['lstm']=='True':
                            _ = reset_ep_lstm_actor(action_net, [action_net.layers[3]], ind)
                        ep_return = infos[ind].get('episode').get('r')
                        returns.append(ep_return)
                        episodes += 1
                        S1 += ep_return
                        S2 += ep_return ** 2
                        avg = S1 / episodes
                        var = S2 / episodes - avg ** 2
                        prob = var / (episodes * a ** 2)
                        print(f'Avg = {avg}, Var = {var}, Pr(X-Avg > {a}) = {prob}')

                    start_time = np.random.rand() * (env.velf.t_range * 1 / 4 - 5.1) + env.velf.t_range * 3 / 4
                    envs.envs[ind].particle.current_t = start_time
                    next_state[ind, :] = env.pos_to_state(envs.envs[ind].particle.current_x,
                                                          envs.envs[ind].particle.current_y,
                                                          envs.envs[ind].particle.current_t)

            next_state = scale_state(next_state, scaler)

        np.save(file_name + '/eval_returns.npy', returns)



    elif info['QL']=='True':
        Q = np.load(file_name + '/Q_Matrix.npy')
        intervals = np.load(file_name + '/Intervals.npy')
        scaler = load_scaler(file_name)

        returns = []
        episodes = 0
        S1 = 0.0
        S2 = 0.0
        prob = 100.0
        a = 0.1
        while prob >= 0.05 or episodes <= 100:
            #     env.particle.current_t = start_time
            state = get_discrete_state_custom(scale_state(env.reset(), scaler).numpy().reshape(-1), intervals)
            done = False
            while not done:
                action = np.argmax(Q[state, :])
                if three_D:
                    env_action = get_env_action3D(action)
                else:
                    env_action = get_env_action2D(action)
                next_env_state, reward, done, _ = env.step(env_action)
                state = get_discrete_state_custom(scale_state(next_env_state, scaler).numpy().reshape(-1), intervals)

            ep_return = env.particle.z_distance_travelled(env.target_dir)
            returns.append(ep_return)
            episodes += 1
            S1 += ep_return
            S2 += ep_return ** 2
            avg = S1 / episodes
            var = S2 / episodes - avg ** 2
            prob = var / (episodes * a ** 2)

            if episodes % 100:
                print('Avg = ' + '%.3f' % avg + ', Var = ' + '%.3f' % var + f', Pr(X-Avg > {a}) = ' + '%.3f' % prob)

        np.save(file_name + '/eval_returns.npy', returns)

    else:
        raise RuntimeError("Did not specify the learning algorithm")
