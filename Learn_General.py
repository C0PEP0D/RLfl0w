from Scripts.Field import *
from Scripts.Gym_Envs import *
from Scripts.Useful import *
from Scripts.Algorithms import *
from stable_baselines3.common.env_util import make_vec_env

import argparse
import os
from distutils.util import strtobool

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    # Flow specific arguments:
    parser.add_argument("--TGV", action="store_true", help="Learn on TGV flow")
    parser.add_argument("--ABC", action="store_true", help="Learn on ABC flow")
    parser.add_argument("--turb", action="store_true", help="Learn on 2D turbulence")

    # Particle specific arguments:
    parser.add_argument("--uswim", type=float, required=True, help="The agent's swimming velocity")
    parser.add_argument("--dir", nargs='+', type=float, help="Target direction of the agent")
    parser.add_argument("--pedley", type=float, default=0.0, help="The agent's reorientation time")

    # Environment specific arguments:
    parser.add_argument("--vort", action="store_true", help="Agent has access to the local vorticity of the flow")
    parser.add_argument("--grad", action="store_true", help="Agent has access to the local velocity gradients of the flow")
    parser.add_argument("--vel", action="store_true", help="Agent has access to the local velocity of the flow")
    parser.add_argument("--relgrad", action="store_true", help="Agent has access to the relative local velocity gradients of the flow")
    parser.add_argument("--projgrad", action="store_true", help="Agent has access to the projected local velocity gradients of the flow")
    parser.add_argument("--tau", action="store_true", help="Learn the tau parameter of the surf")
    parser.add_argument("--time_step", type=float, default=0.01, help="The environment time step")
    parser.add_argument("--N_steps", type=int, default=40, help="The environment number of action steps")
    parser.add_argument("--env_steps", type=int, required=True, default=100, help="The environment steps executed per action")

    # Algorithm specific arguments:
    parser.add_argument("--A2C", action="store_true", help="Use A2C as learning algorithm")
    parser.add_argument("--PPO", action="store_true", help="Use PPO as learning algorithm")
    parser.add_argument("--QL", action="store_true", help="Use Q_learning as learning algorithm")


    #A2C:
    parser.add_argument("--lr_actor", type=float, default=1e-6, help="the learning rate of the actor optimizer")
    parser.add_argument("--lr_critic", type=float, default=1e-4, help="the learning rate of the critic optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--num_episodes", type=int, default=100000, help="Number of learning episodes")

    # Q_Learning
    parser.add_argument("--lr", type=float, default=0.8, help="the learning rate for Q_learning")
    parser.add_argument("--epsilon", type=float, default=0.1, help="the exploration rate for Q_learning")
    parser.add_argument("--num_discrete_states", type=int, default=3, help="the number of discrete states per variable")

    #PPO
    parser.add_argument("--num_envs", type=int, default=10, help="the number of parallel game environments")
    parser.add_argument("--steps_per_update", type=int, default=10, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal_lr", action="store_true", help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae",  action="store_true", help="Use GAE for advantage computation")
    parser.add_argument("--gae_lambda", type=float, default=1.0, help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=5, help="the number of mini-batches")
    parser.add_argument("--learn_step", type=int, default=1, help="number of steps to skip while training")
    parser.add_argument("--update_epochs", type=int, default=4, help="the K epochs to update the policy")
    parser.add_argument("--clip_coef", type=float, default=0.1, help="the surrogate clipping coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--target_kl", type=float, default=0.02, help="the target KL divergence threshold")
    parser.add_argument("--lstm", action="store_true", help="Use Long Short Term Memory policy for both actor and critic networks")

    # Saving and evaluating arguments:
    parser.add_argument("--file_name", type=str, required=True, help="the file name to save everything")
    parser.add_argument("--save_frequency", type=int, default=10000, help="saving frequency of the neural network and info")
    parser.add_argument("--lr_curve", type=lambda x: bool(strtobool(x)), default=True, help="saving frequency of the neural network and info")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=True, help="print some information while learning")


    # Scaler specific arguments:
    parser.add_argument("--scaler_states", type=int, default=100000, help="the number of statest used to scale the states")

    # Model specific arguments:
    parser.add_argument("--cyclic", type=lambda x: bool(strtobool(x)), default=True, help="the policy is cyclic or not if the output is an angle it is cyclic")


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    parsed_dict = vars(args)
    three_D = False



    root = './New_Models/'
    file_name = os.path.join(root, args.file_name)
    os.mkdir(file_name)

    save_parse(file_name, parsed_dict)


    if args.TGV:
        VelF = VelField(2 * np.pi, 2 * np.pi, 50, 50)
        VelF.Generate_TGV()
        Uswim = args.uswim
        dir_particle = args.dir
        ActivePart = ActiveParticle(1.0, 1.0, VelF, np.pi / 2, Uswim)
        if args.vort:
            env = VortEnv2D(Particle=ActivePart, Velocity_Field=VelF, Direction=dir_particle, time_step=args.time_step,
                            N_steps=args.N_steps, env_steps=args.env_steps)
            envs = make_vec_env(VortEnv2D, n_envs=args.num_envs,
                                env_kwargs={"Velocity_Field": VelF, "Particle": ActivePart, "Direction": dir_particle,
                                            "time_step": args.time_step,
                                            "N_steps": args.N_steps, "env_steps": args.env_steps})
        elif args.grad:
            env = GradEnv2D(Particle=ActivePart, Velocity_Field=VelF, Direction=dir_particle, time_step=args.time_step,
                            N_steps=args.N_steps, env_steps=args.env_steps)
            envs = make_vec_env(GradEnv2D, n_envs=args.num_envs,
                                env_kwargs={"Velocity_Field": VelF, "Particle": ActivePart, "Direction": dir_particle,
                                            "time_step": args.time_step,
                                            "N_steps": args.N_steps, "env_steps": args.env_steps})
        else:
            raise RuntimeError("Did not specify an environment")

    elif args.ABC:
        VelF = VelField3D(2 * np.pi, 2 * np.pi, 2 * np.pi, 50, 50, 50)
        three_D = True
        A = np.sqrt(3)
        B = np.sqrt(2)
        C = np.float64(1.0)
        VelF.Generate_ABC(A, B, C)
        Uswim = args.uswim
        dir_particle = args.dir
        ActivePart = ActiveParticle3D(1.0, 1.0, 1.0, VelF, np.pi / 4, np.pi / 4, Uswim)
        if args.vort:
            env = VortEnv3D(Particle=ActivePart, Velocity_Field=VelF, Direction=dir_particle, time_step=args.time_step,
                            N_steps=args.N_steps, env_steps=args.env_steps)
            envs = make_vec_env(VortEnv3D, n_envs=args.num_envs,
                                env_kwargs={"Velocity_Field": VelF, "Particle": ActivePart, "Direction": dir_particle,
                                            "time_step": args.time_step,
                                            "N_steps": args.N_steps, "env_steps": args.env_steps})
        elif args.grad:
            env = VortEnv3D(Particle=ActivePart, Velocity_Field=VelF, Direction=dir_particle, time_step=args.time_step,
                            N_steps=args.N_steps, env_steps=args.env_steps)
            envs = make_vec_env(VortEnv3D, n_envs=args.num_envs,
                                env_kwargs={"Velocity_Field": VelF, "Particle": ActivePart, "Direction": dir_particle,
                                            "time_step": args.time_step,
                                            "N_steps": args.N_steps, "env_steps": args.env_steps})
        else:
            raise RuntimeError("Did not specify an environment")

    elif args.turb:
        n = 256
        EMISSION_RATE = 1.0
        LIFETIME = 5.0
        DELTA_TIMEIT = 50
        MAX_TIME = 1 * DELTA_TIMEIT
        DT = 1e-3
        N_FRAMES = int(MAX_TIME / (DELTA_TIMEIT * DT))
        u, v = load_velocity(n, EMISSION_RATE, LIFETIME, DELTA_TIMEIT, MAX_TIME, DT)
        VelF = VelFieldTran(2 * np.pi, 2 * np.pi, MAX_TIME, n, n, N_FRAMES, u, v)
        Uswim = args.uswim
        dir_particle = args.dir
        ActivePart = ActiveParticleTran(5.1, 2.5, 0.0, VelF, np.pi / 2, Uswim, pedley=args.pedley)
        if args.vort:
            env = GradEnvTran(Particle=ActivePart, Velocity_Field=VelF, Direction=dir_particle, time_step=args.time_step,
                            N_steps=args.N_steps, env_steps=args.env_steps)
            envs = make_vec_env(GradEnvTran, n_envs=args.num_envs,
                                env_kwargs={"Velocity_Field": VelF, "Particle": ActivePart, "Direction": dir_particle,
                                            "time_step": args.time_step,
                                            "N_steps": args.N_steps, "env_steps": args.env_steps})
        elif args.grad:
            env = GradEnvTran(Particle=ActivePart, Velocity_Field=VelF, Direction=dir_particle, time_step=args.time_step,
                            N_steps=args.N_steps, env_steps=args.env_steps)
            envs = make_vec_env(GradEnvTran, n_envs=args.num_envs,
                                env_kwargs={"Velocity_Field": VelF, "Particle": ActivePart, "Direction":dir_particle,
                                            "time_step": args.time_step,
                                            "N_steps": args.N_steps, "env_steps": args.env_steps})

        elif args.vel:
            env = VelEnvTran(Particle=ActivePart, Velocity_Field=VelF, Direction=dir_particle, time_step=args.time_step,
                            N_steps=args.N_steps, env_steps=args.env_steps)
            envs = make_vec_env(VelEnvTran, n_envs=args.num_envs,
                                env_kwargs={"Velocity_Field": VelF, "Particle": ActivePart, "Direction":dir_particle,
                                            "time_step": args.time_step,
                                            "N_steps": args.N_steps, "env_steps": args.env_steps})


        else:
            raise RuntimeError("Did not specify an environment")

    else:
        raise RuntimeError("Did not specify a flow to learn on!")

    if args.A2C or args.PPO:
        input_dims = env.observation_space.shape[0]
        lr_actor = args.lr_actor
        lr_critic = args.lr_critic
        if args.lstm:
            value_net = value_function_lstm(input_dims, args.num_envs)
        else:
            value_net = value_function(input_dims)
        optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)

    gamma = args.gamma
    num_episodes = args.num_episodes
    scaler, state_space_samples = build_scaler(env, args.scaler_states)
    if args.scaler_states == 1:
        scaler.mean_ = np.zeros(scaler.scale_.shape)
        scaler.var_ = np.ones(scaler.scale_.shape)
    save_scaler(scaler, file_name)

    if args.A2C:
        if three_D:
            action_net = policy_network3D(input_dims)
        else:
            action_net = policy_network(input_dims, cyclic=args.cyclic)
        model = A2C(env, action_net, value_net, optimizer_actor, optimizer_critic, scaler, gamma, three_D=three_D)
        model.learn(num_episodes, file_name=file_name, save_freq=args.save_frequency, Verbose=args.verbose, lr_curve=args.lr_curve)
    elif args.PPO:

        if args.lstm:
            action_net = policy_network_lstm(input_dims, args.num_envs, conversion=get_sample_and_logprob_lstm, cyclic=args.cyclic)
        else:
            if three_D:
                action_net = policy_network3D(input_dims, conversion=get_sample_and_logprob)
            else:
                action_net = policy_network(input_dims, conversion=get_sample_and_logprob, cyclic=args.cyclic)


        gae_lambda = args.gae_lambda
        ent_coef = args.ent_coef
        target_kl = args.target_kl
        clip_coef = args.clip_coef
        epochs = args.update_epochs
        steps_per_update = args.steps_per_update
        num_mini_batches = args.num_minibatches
        GAE = bool(args.gae)
        anneal_lr = bool(args.anneal_lr)

        if args.lstm:
            lstm_layers_actor = [action_net.layers[2]]
            lstm_layers_critic = [value_net.layers[2]]
            model = PPO_LSTM(envs, action_net, value_net, lstm_layers_actor, lstm_layers_critic, optimizer_actor,
                             optimizer_critic,lr_actor, lr_critic, scaler, gamma, gae_lambda, ent_coef, target_kl,
                             clip_coef, epochs, steps_per_update, num_mini_batches, GAE=GAE, anneal_lr=anneal_lr,
                             learned_episodes=0, one_opt=False)

            model.learn(num_episodes, file_name=file_name, save_freq=args.save_frequency, Verbose=args.verbose, lr_curve=args.lr_curve)
        else:
            model = PPO(envs, action_net, value_net, optimizer_actor, optimizer_critic, lr_actor, lr_critic, scaler, gamma,
                        gae_lambda, ent_coef, target_kl, clip_coef, epochs, steps_per_update, num_mini_batches, GAE=GAE,
                        anneal_lr=anneal_lr, learned_episodes=0, one_opt=False, learn_step=args.learn_step)

            model.learn(num_episodes, file_name=file_name, save_freq=args.save_frequency, Verbose=args.verbose, lr_curve=args.lr_curve)

    elif args.QL:
        learning_rate = args.lr
        states_per_variable = args.num_discrete_states
        epsilon = args.epsilon
        model = Q_Learning( env, learning_rate, states_per_variable, state_space_samples, epsilon, scaler, gamma,
             three_D=three_D)
        model.learn(num_episodes, file_name=file_name, save_freq=args.save_frequency, Verbose=args.verbose,
                    lr_curve=args.lr_curve)

    else:
        raise RuntimeError("Did not specify the learning algorithm")
