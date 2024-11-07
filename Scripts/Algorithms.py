from Scripts.Models import *
from Scripts.Useful import Evaluate_Policy
from Scripts.Plotting import *
from Scripts.Saving_Loading import *
from scipy import stats
from bisect import bisect


class A2C:
    def __init__(self, env, action_net, value_net, optimizer_actor, optimizer_critic, scaler, gamma,
                 learned_episodes=0, three_D=False):
        self.three_D = three_D
        self.learned_episodes = learned_episodes
        self.name = 'A2C'
        self.env = env
        self.action_net = action_net
        self.value_net = value_net
        self.lazy_action_net = tf.function(action_net)
        self.lazy_value_net = tf.function(value_net)
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.scaler = scaler
        self.gamma = gamma
        self.loss_critic = tf.keras.losses.MeanSquaredError()
        self.eval_steps = []
        self.eval_returns = []

    def learn(self, num_episodes, file_name=False, save_freq=100000, eval_freq=0, eval_ep=0, Verbose=True, lr_curve=False):
        episode_history = []
        avg_ep_history = []
        for episode in range(num_episodes):
            state = self.env.reset()
            steps = 0
            done = False
            while not done:
                # if self.three_D:
                #     action = self.lazy_action_net(scale_state(state, self.scaler))
                #     act = np.squeeze(action).reshape((3,))
                #     theta = np.arccos(act[2])
                #     phi = np.arctan2(act[1], act[0])
                #     act = [theta / np.pi, phi / np.pi]
                # else:
                #     action = self.lazy_action_net(scale_state(state, self.scaler))
                #     if self.env.type == 'Surf':
                #         act = np.squeeze(action, axis=0).reshape((1,))
                #         # if act[0] <= 0.0:
                #         #     act = np.array([0.0])
                #         # if act[0] >= 10.0:
                #         #     act = np.array([10.0])
                #     else:
                #         act = np.squeeze(action / np.pi, axis=0).reshape((1,))
                action = self.lazy_action_net(scale_state(state, self.scaler))
                act = np.squeeze(action).reshape(self.env.action_space.shape)
                next_state, rew, done, _ = self.env.step(act)
                reward = tf.convert_to_tensor([[np.float32(rew)]])
                steps += 1


                v_of_next_state = self.lazy_value_net(scale_state(next_state, self.scaler))
                target = reward + self.gamma * v_of_next_state

                v_of_state = train_value_net(self.value_net, scale_state(state, self.scaler),
                                             target, self.optimizer_critic, self.loss_critic)

                train_action_net(self.action_net, scale_state(state, self.scaler),
                                 target - v_of_state, action, self.optimizer_actor)
                state = next_state

            episode_history.append(self.env.particle.z_distance_travelled(self.env.target_dir))
            self.learned_episodes += 1
            if episode>=999:
                current_avg = sum(episode_history[-1000:])/1000
                avg_ep_history.append(current_avg)
            else:
                current_avg = sum(episode_history) / len(episode_history)
                avg_ep_history.append(current_avg)

            cond = False
            if Verbose:
                print("Episode: {}, Cumulative reward: {:0.2f}".format(
                    episode, episode_history[-1]))

            if file_name:
                if not ((episode + 1) % save_freq):
                    print("SAVING MODEL")
                    cond = self.save(file_name)
            if file_name:
                if eval_freq:
                    if not ((episode + 1) % eval_freq):
                        print("EVALUATING MODEL")
                        self.evaluate(eval_ep, file_name)
            if file_name:
                if lr_curve and episode%100==0:
                        plot_lr_curve(np.arange(len(avg_ep_history)), avg_ep_history, file_name)
            if cond:
                break

    def save(self, file_name):
        done= False
        save_optimizers(file_name, self.optimizer_actor, self.optimizer_critic)
        save_network_weights(file_name, self.value_net, self.action_net)
        info = {'input': self.env.type, 'input_dim': self.env.observation_space.shape[0],
                'number_of_Hidden_Layers_Actor': 2,
                'size_of_Hidden_Layers_Actor': 100, 'Gamma': self.gamma,
                'lr_critic_now': self.optimizer_critic.learning_rate.numpy(),
                'lr_actor_now': self.optimizer_actor.learning_rate.numpy(),
                'Episode_Length': self.env.N_steps, 'Delta_t': self.env.dt,
                'Env_Steps': self.env.env_steps,
                'Number_of_Episodes': self.learned_episodes, 'V_Swim': self.env.particle.vswim,
                'Target_dir': self.env.target_dir, 'Learning_Algorithm_Name': self.name}

        save_info_general(file_name, info)

        return done


    def evaluate(self, eval_episodes, file_name):
        average_reward = Evaluate_Policy(self.env, 1.0, eval_episodes, self.scaler, Random_Policy=False,
                                         Deterministic=False,
                                         three_D=self.three_D,
                                         env_steps=1,
                                         actor=self.lazy_action_net)

        self.eval_steps.append(self.learned_episodes)
        self.eval_returns.append(average_reward)
        save_learning_curve(self.eval_steps, self.eval_returns, file_name)

        fig, ax = Plot_Learning_Curve(self.eval_steps, self.eval_returns)
        fig.savefig(file_name + '/Learning_Curve.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig)


    def obtain_gradients(self):
        state = self.env.reset()
        steps = 0
        counter = 0
        done = False
        while not done:
            act = self.lazy_action_net(scale_state(state, self.scaler))
            if self.env.type == 'Surf':
                action = act
            else:
                action = act / np.pi
            reward = 0
            for i in range(self.env_steps):
                next_state, rew, done, _ = self.env.step(np.squeeze(action, axis=0).reshape((1,)))
                rew = tf.convert_to_tensor([[np.float32(rew)]])
                reward += rew
                steps += 1
                if done:
                    break

            v_of_next_state = self.lazy_value_net(scale_state(next_state, self.scaler))
            target = reward + self.gamma * v_of_next_state

            v_of_state, value_grads = gradients_value_net(self.value_net, scale_state(state, self.scaler),
                                                          target, self.loss_critic)

            actor_grads = gradients_action_net(self.action_net, scale_state(state, self.scaler),
                                               target - v_of_state, act)

            if counter == 0:
                sum_value_grads = value_grads
                sum_actor_grads = actor_grads
            else:
                sum_value_grads = [tf.math.add(value_grads[i], j) for i, j in enumerate(sum_value_grads)]
                sum_actor_grads = [tf.math.add(actor_grads[i], j) for i, j in enumerate(sum_actor_grads)]
            counter += 1
            state = next_state

        return [sum_value_grads, sum_actor_grads]

    def learn_batched(self, num_episodes, file_name=False, save_freq=100000, eval_freq=0, eval_ep=0, Verbose=True):
        episode_history = []
        for episode in range(num_episodes):
            critic_grads, actor_grads = self.obtain_gradients()
            self.learned_episodes += 1
            episode_history.append(self.env.particle.z_distance_travelled(self.env.target_dir))
            if Verbose:
                print("Episode: {}, Cumulative reward: {:0.2f}".format(
                    self.learned_episodes, episode_history[-1]))
            if file_name:
                if not (self.learned_episodes % save_freq):
                    print("SAVING MODEL")
                    self.save(file_name)
                if eval_freq:
                    if not (self.learned_episodes % eval_freq):
                        print("EVALUATING MODEL")
                        self.evaluate(eval_ep, file_name)
            self.optimizer_critic.apply_gradients(zip(critic_grads, self.value_net.trainable_variables))
            self.optimizer_actor.apply_gradients(zip(actor_grads, self.action_net.trainable_variables))


class PPO:
    def __init__(self, envs, action_net, value_net, optimizer_actor, optimizer_critic, lr_actor, lr_critic, scaler, gamma,
                 gae_lambda, ent_coef, target_kl, clip_coef, epochs, M, num_mini_batches, GAE = False,
                 anneal_lr = False, learned_episodes = 0, one_opt = False, learn_step = 1):
        self.name = 'PPO'
        self.one_opt = one_opt
        self.envs = envs
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.learned_episodes = learned_episodes
        self.N = envs.num_envs
        self.epochs = epochs
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.GAE = GAE
        self.gae_lambda = gae_lambda
        self.M = M
        self.learn_step = learn_step
        self.batch_size = int(M * self.N / self.learn_step)
        self.num_mini_batches = num_mini_batches
        self.clip_coef = clip_coef
        self.mini_batch_size = int(self.batch_size // num_mini_batches)

        self.action_net = action_net
        self.value_net = value_net
        self.lazy_action_net = tf.function(action_net)
        self.lazy_value_net = tf.function(value_net)
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.scaler = scaler

    def learn(self, num_episodes, file_name=False, save_freq=100000, Verbose=True, lr_curve=False):
        total_timesteps = int(self.envs.envs[0].N_steps * num_episodes)
        steps = total_timesteps // (self.N * self.M)

        obs = np.zeros((self.M, self.N) + self.envs.observation_space.shape)
        next_obs = np.zeros((self.M, self.N) + self.envs.observation_space.shape)
        actions = np.zeros((self.M, self.N) + self.envs.action_space.shape)
        logprobs = np.zeros((self.M, self.N))
        rewards = np.zeros((self.M, self.N))
        dones = np.zeros((self.M, self.N))
        values = np.zeros((self.M + 1, self.N))
        next_values = np.zeros((self.M, self.N))
        last_done = np.zeros((self.N,))
        list_clip_ratio = []
        list_KL_divergence = []

        next_state = scale_state(self.envs.reset(), self.scaler)
        next_done = [False for i in range(self.N)]

        current_avg = 0.0
        ep_history = []
        avg_ep_history = []
        episodes = []
        # Update Loop
        for update in range(0, steps):
            # Anneal the Learning Rate
            if self.anneal_lr:
                frac = 1.0 - update / steps
                lrnow_actor = frac * self.lr_actor
                lrnow_critic = frac * self.lr_critic
                self.optimizer_actor.lr.assign(lrnow_actor)
                self.optimizer_critic.lr.assign(lrnow_critic)

            values[0] = self.lazy_value_net(next_state).numpy().reshape(self.N)
            # Acquire Training samples
            for step in range(0, self.M):
                state = next_state
                obs[step] = state
                dones[step] = next_done
                action_logprob = self.lazy_action_net(state)
                actions[step] = action_logprob[:, 0:self.envs.action_space.shape[0]].numpy().reshape(
                    (self.N,) + self.envs.action_space.shape)
                logprobs[step] = action_logprob[:, -1].numpy()
                next_state, reward, next_done, info = self.envs.step(action_logprob[:, :-1].numpy())
                next_state = scale_state(next_state, self.scaler)
                rewards[step] = reward
                if any(next_done):
                    for ind, elem in enumerate(next_done):
                        if elem == True:
                            next_obs[step, ind] = scale_state(info[ind].get('terminal_observation'), self.scaler)
                            ep_history.append(info[ind].get('episode').get('r'))
                            self.learned_episodes += 1
                            next_values[step, ind] = self.lazy_value_net(
                                next_obs[step, ind].reshape((1, -1))).numpy().reshape(-1)
                            values[step + 1, ind] = self.lazy_value_net(
                                tf.reshape(next_state[ind], [1, -1])).numpy().reshape(-1)
                        else:
                            next_obs[step, ind] = next_state[ind]
                            next_values[step, ind] = self.lazy_value_net(next_obs[step, ind]).numpy().reshape(-1)
                            values[step + 1, ind] = next_values[step, ind]
                else:
                    next_obs[step] = next_state
                    next_values[step] = self.lazy_value_net(next_obs[step]).numpy().reshape(self.N)
                    values[step + 1] = next_values[step]

            last_done[:] = next_done

            # Calculate Advantages

            # Generalized Advantage Estimation
            if self.GAE:
                advantages = np.zeros((self.M, self.N))
                lastgaelam = 0
                for t in reversed(range(self.M)):
                    nextvalues = next_values[t]
                    if t == self.M - 1:
                        nextnonterminal = 1.0 - last_done
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

                returns = advantages + values[:-1, :]

            # Lambda=1
            else:
                returns = np.zeros((self.M, self.N))
                for t in reversed(range(self.M)):
                    if t == self.M - 1:
                        nextnonterminal = 1.0 - last_done
                        next_return = next_values[t]
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + self.gamma * nextnonterminal * next_return + self.gamma * (1.0 - nextnonterminal) * \
                                 next_values[t]

                advantages = returns - values[:-1, :]

            # Flatten Batches and Convert to Tensors

            tf_obs = tf.convert_to_tensor(obs[0::self.learn_step,:,:].reshape((-1,) + self.envs.observation_space.shape))
            tf_actions = tf.convert_to_tensor(actions[0::self.learn_step,:,:].reshape((-1,) + self.envs.action_space.shape))
            tf_logprobs = tf.convert_to_tensor(logprobs[0::self.learn_step,:].reshape((-1,)))
            tf_returns = tf.convert_to_tensor(returns[0::self.learn_step,:].reshape((-1,)))
            tf_advantages = tf.convert_to_tensor(advantages[0::self.learn_step,:].reshape((-1,)))

            # Optimizing the policy and value network

            b_inds = np.arange(self.batch_size)
            clipfracs = []
            approx_kl = 0.0
            for epoch in range(self.epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.mini_batch_size):
                    end = start + self.mini_batch_size

                    # Setup Mini-Batch Samples
                    mb_inds = b_inds[start:end]
                    mb_obs = tf.gather(tf_obs, indices=mb_inds)
                    mb_actions = tf.gather(tf_actions, indices=mb_inds)
                    mb_logprobs = tf.gather(tf_logprobs, indices=mb_inds)
                    mb_advantages = tf.cast(tf.gather(tf_advantages, indices=mb_inds), dtype=tf.float32)
                    mb_returns = tf.cast(tf.gather(tf_returns, indices=mb_inds), dtype=tf.float32)

                    # Train Actor
                    if not self.one_opt:
                        if self.ent_coef:
                            logratio, ratio = train_action_ppo(self.action_net, mb_obs, tf.constant(self.clip_coef),
                                                               tf.constant(self.ent_coef), mb_actions, mb_advantages,
                                                               mb_logprobs, self.optimizer_actor)
                        else:
                            logratio, ratio = train_action_ppo_noent(self.action_net, mb_obs, tf.constant(self.clip_coef), mb_actions,
                                                                     mb_advantages, mb_logprobs, self.optimizer_actor)
                    else:
                        if self.ent_coef:
                            logratio, ratio = train_everything_ppo(self.action_net, self.value_net, mb_obs,
                                                                   tf.constant(self.clip_coef),
                                                                   tf.constant(self.ent_coef), mb_actions, mb_advantages,
                                                                   mb_returns, mb_logprobs, self.optimizer_actor)
                        else:
                            logratio, ratio = train_everything_ppo_noent(self.action_net, self.value_net, mb_obs,
                                                                         tf.constant(self.clip_coef),
                                                                         mb_actions, mb_advantages, mb_returns,
                                                                         mb_logprobs,
                                                                         self.optimizer_actor)

                    # Calculate KL divergence and clipfracs
                    old_approx_kl = tf.math.reduce_mean(-logratio)
                    approx_kl = tf.math.reduce_mean((ratio - 1.0) - logratio)
                    clipfracs += [
                        tf.math.reduce_mean(tf.cast((tf.math.abs(ratio - 1.0) > self.clip_coef), dtype=tf.float32))]

                    # Train Critic
                    if not self.one_opt:
                        train_value_ppo(self.value_net, mb_obs, mb_returns, self.optimizer_critic)

                    # Consider removing that one
                    if self.target_kl is not None:
                        if approx_kl > self.target_kl:
                            break

                # Check Divergence with Original Policy
                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

            # Build Episodic Return
            if len(ep_history) >= 1000:
                current_avg = sum(ep_history[-1000:]) / 1000
                avg_ep_history.append(current_avg)
                episodes.append(self.learned_episodes)

            elif ep_history:
                current_avg = sum(ep_history) / len(ep_history)
                avg_ep_history.append(current_avg)
                episodes.append(self.learned_episodes)

            # Print Some Learning Parameters
            if Verbose:
                print(f'UPDATE {update}:')
                print(f'Clip Fraction: {np.mean(clipfracs)} || KL Divergence: {approx_kl.numpy()} || Epochs Learned: {epoch} || Current Average Return: {current_avg}')
                list_clip_ratio.append(np.mean(clipfracs))
                list_KL_divergence.append(approx_kl.numpy())
                save_clipfrac_divergence(list_clip_ratio, list_KL_divergence, file_name)

            if file_name:
                if lr_curve and len(avg_ep_history)%100==0:
                    plot_lr_curve(episodes, avg_ep_history, file_name)


                if not ((update + 1) % save_freq):
                    print("SAVING MODEL")
                    self.save(file_name)


    def save(self, file_name):
        save_optimizers(file_name, self.optimizer_actor, self.optimizer_critic)
        save_network_weights(file_name, self.value_net, self.action_net)

        info = {'input': self.envs.envs[0].type, 'input_dim': self.envs.observation_space.shape[0], 'number_of_Hidden_Layers_Actor': 2,
                'size_of_Hidden_Layers_Actor': 100, 'Gamma': self.gamma, 'lr_critic': self.lr_critic, 'lr_actor': self.lr_actor,
                'lr_critic_now': self.optimizer_critic.learning_rate.numpy(), 'lr_actor_now': self.optimizer_actor.learning_rate.numpy(),
                'Episode_Length': self.envs.envs[0].N_steps, 'Delta_t': self.envs.envs[0].dt, 'Env_Steps': self.envs.envs[0].env_steps,
                'Number_of_Episodes': self.learned_episodes, 'V_Swim': self.envs.envs[0].particle.vswim,
                'Target_dir': self.envs.envs[0].target_dir,
                'Number_of_Environments': self.N, 'Number_of_Steps_per_Env_per_Update': self.M, 'Number_of_Epochs_per_Update': self.epochs,
                'Number_of_Minibatches_per_Update': self.num_mini_batches, 'PPO_Clip_Coef': self.clip_coef, 'PPO_Entropy_Coef': self.ent_coef,
                'PPO_GAE_Lambda': self.gae_lambda, 'Learning_Algorithm_Name': self.name, 'Using_One_Optimizer': self.one_opt}

        save_info_general(file_name, info)


class PPO_LSTM:
    def __init__(self, envs, action_net, value_net, lstm_layers_actor, lstm_layers_critic, optimizer_actor,
                 optimizer_critic, lr_actor, lr_critic, scaler, gamma,
                 gae_lambda, ent_coef, target_kl, clip_coef, epochs, M, num_mini_batches, GAE=False,
                 anneal_lr=False, learned_episodes=0, one_opt=False):
        self.name = 'PPO'
        self.one_opt = one_opt
        self.envs = envs
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.learned_episodes = learned_episodes
        self.N = envs.num_envs
        self.epochs = epochs
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.GAE = GAE
        self.gae_lambda = gae_lambda
        self.M = M
        self.batch_size = int(M * self.N)
        self.num_mini_batches = num_mini_batches
        self.clip_coef = clip_coef
        self.mini_batch_size = int(self.batch_size // num_mini_batches)

        self.action_net = action_net
        self.value_net = value_net
        self.lstm_layers_actor = lstm_layers_actor
        self.lstm_layers_critic = lstm_layers_critic
        self.actor_lstm_shapes = [(layer.output_shape[0], layer.output_shape[-1]) for layer in self.lstm_layers_actor]
        self.critic_lstm_shapes = [(layer.output_shape[0], layer.output_shape[-1]) for layer in self.lstm_layers_critic]

        self.lazy_action_net = tf.function(action_net)
        self.lazy_value_net = tf.function(value_net)
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.scaler = scaler

    def learn(self, num_episodes, file_name=False, save_freq=100000, Verbose=True, lr_curve=False):
        total_timesteps = int(self.envs.envs[0].N_steps * num_episodes)
        steps = total_timesteps // (self.N * self.M)

        obs = np.zeros((self.M, self.N) + self.envs.observation_space.shape)
        next_obs = np.zeros((self.M, self.N) + self.envs.observation_space.shape)
        actions = np.zeros((self.M, self.N) + self.envs.action_space.shape)
        logprobs = np.zeros((self.M, self.N))
        rewards = np.zeros((self.M, self.N))
        dones = np.zeros((self.M, self.N))
        values = np.zeros((self.M + 1, self.N))
        next_values = np.zeros((self.M, self.N))
        last_done = np.zeros((self.N,))
        next_lstm_states_actor = self.get_lstm_actor()
        next_lstm_states_critic = self.get_lstm_critic()

        next_state = scale_state(self.envs.reset(), self.scaler)
        next_done = [False for i in range(self.N)]

        current_avg = 0.0
        ep_history = []
        avg_ep_history = []
        episodes = []
        # Update Loop
        for update in range(0, steps):
            # Anneal the Learning Rate
            if self.anneal_lr:
                frac = 1.0 - update / steps
                lrnow_actor = frac * self.lr_actor
                lrnow_critic = frac * self.lr_critic
                self.optimizer_actor.lr.assign(lrnow_actor)
                self.optimizer_critic.lr.assign(lrnow_critic)

            initial_lstm_states_critic = self.reset_val_lstm_critic(next_lstm_states_critic)
            initial_lstm_states_actor = self.reset_val_lstm_actor(next_lstm_states_actor)
            values[0] = self.lazy_value_net(tf.expand_dims(next_state, axis=1)).numpy().reshape(self.N)
            reset_times = []
            reset_inds = []
            reset_lstm_actor_states = []
            reset_lstm_critic_states = []
            # Acquire Training samples
            for step in range(0, self.M):
                state = next_state
                obs[step] = state
                dones[step] = next_done
                action_logprob = tf.squeeze(self.lazy_action_net(tf.expand_dims(state, axis=1)), axis=1)
                actions[step] = action_logprob[:, 0:self.envs.action_space.shape[0]].numpy().reshape(
                    (self.N,) + self.envs.action_space.shape)
                logprobs[step] = action_logprob[:, -1].numpy()
                next_state, reward, next_done, info = self.envs.step(action_logprob[:, :-1].numpy())
                next_state = scale_state(next_state, self.scaler)
                rewards[step] = reward
                if any(next_done):
                    reset_times.append(step)
                    reset_indices = []
                    for ind, elem in enumerate(next_done):
                        if elem == True:
                            reset_indices.append(ind)
                            next_obs[step, ind] = scale_state(info[ind].get('terminal_observation'), self.scaler)
                            ep_history.append(info[ind].get('episode').get('r'))
                            self.learned_episodes += 1
                        else:
                            next_obs[step, ind] = next_state[ind]
                    reset_inds.append(reset_indices)
                    next_values[step] = self.lazy_value_net(tf.expand_dims(next_obs[step], axis=1)).numpy().reshape(self.N)

                    for ind in reset_indices:
                        saved_actor_states = self.reset_ep_lstm_actor(ind)
                        saved_critic_states = self.reset_ep_lstm_critic(ind)

                    reset_lstm_actor_states.append(saved_actor_states)
                    reset_lstm_critic_states.append(saved_critic_states)
                    next_lstm_states_critic = self.get_lstm_critic()

                    values[step + 1] = self.lazy_value_net(tf.expand_dims(next_state, axis=1)).numpy().reshape(-1)



                else:
                    next_obs[step] = next_state
                    next_lstm_states_critic = self.get_lstm_critic()
                    next_values[step] = self.lazy_value_net(tf.expand_dims(next_obs[step], axis=1)).numpy().reshape(self.N)
                    values[step + 1] = next_values[step]

            next_lstm_states_actor = self.get_lstm_actor()
            last_done[:] = next_done

            # Calculate Advantages
            # Generalized Advantage Estimation
            if self.GAE:
                advantages = np.zeros((self.M, self.N))
                lastgaelam = 0
                for t in reversed(range(self.M)):
                    nextvalues = next_values[t]
                    if t == self.M - 1:
                        nextnonterminal = 1.0 - last_done
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

                returns = advantages + values[:-1, :]

            # Lambda=1
            else:
                returns = np.zeros((self.M, self.N))
                for t in reversed(range(self.M)):
                    if t == self.M - 1:
                        nextnonterminal = 1.0 - last_done
                        next_return = next_values[t]
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + self.gamma * nextnonterminal * next_return + self.gamma * (1.0 - nextnonterminal) * \
                                 next_values[t]

                advantages = returns - values[:-1, :]

            # Flatten Batches and Convert to Tensors

            tf_obs = tf.convert_to_tensor(obs.transpose(1,0,2))
            tf_actions = tf.convert_to_tensor(actions.transpose(1,0,2))
            tf_logprobs = tf.convert_to_tensor(logprobs.transpose(1,0))
            tf_returns = tf.convert_to_tensor(returns.transpose(1,0))
            tf_advantages = tf.convert_to_tensor(advantages.transpose(1,0))

            clipfracs = []
            if not reset_times:
                reset_times.append(self.M - 1)
            elif not reset_times[-1] == self.M-1:
                reset_times.append(self.M - 1)
            approx_kl = 0.0
            for epoch in range(self.epochs):

                _ = self.reset_val_lstm_actor(initial_lstm_states_actor)
                _ = self.reset_val_lstm_critic(initial_lstm_states_critic)

                start_time = 0
                for ind, reset_time in enumerate(reset_times):
                    end_time = reset_time+1
                    mb_obs_seq = tf_obs[:,start_time:end_time,:]
                    mb_actions_seq = tf_actions[:,start_time:end_time,:]
                    mb_logprobs_seq = tf_logprobs[:,start_time:end_time]
                    mb_advantages_seq = tf.cast(tf_advantages[:,start_time:end_time], dtype=tf.float32)
                    mb_returns_seq = tf.cast(tf_returns[:,start_time:end_time], dtype=tf.float32)
                    start_time = end_time

                    # Train Actor
                    if not self.one_opt:
                        if self.ent_coef:
                            logratio, ratio = train_action_ppo(self.action_net, mb_obs_seq, tf.constant(self.clip_coef),
                                                               tf.constant(self.ent_coef), mb_actions_seq, mb_advantages_seq,
                                                               mb_logprobs_seq, self.optimizer_actor, lstm=True)
                        else:
                            logratio, ratio = train_action_ppo_noent(self.action_net, mb_obs_seq, tf.constant(self.clip_coef), mb_actions_seq,
                                                                     mb_advantages_seq, mb_logprobs_seq, self.optimizer_actor, lstm=True)
                    else:
                        if self.ent_coef:
                            logratio, ratio = train_everything_ppo(self.action_net, self.value_net, mb_obs_seq,
                                                                   tf.constant(self.clip_coef),
                                                                   tf.constant(self.ent_coef), mb_actions_seq, mb_advantages_seq,
                                                                   mb_returns_seq, mb_logprobs_seq, self.optimizer_actor, lstm=True)
                        else:
                            logratio, ratio = train_everything_ppo_noent(self.action_net, self.value_net, mb_obs_seq,
                                                                         tf.constant(self.clip_coef),
                                                                         mb_actions_seq, mb_advantages_seq, mb_returns_seq,
                                                                         mb_logprobs_seq,
                                                                         self.optimizer_actor, lstm=True)

                    # Calculate KL divergence and clipfracs
                    old_approx_kl = tf.math.reduce_mean(-logratio)
                    approx_kl = tf.math.reduce_mean((ratio - 1.0) - logratio)
                    clipfracs += [
                        tf.math.reduce_mean(tf.cast((tf.math.abs(ratio - 1.0) > self.clip_coef), dtype=tf.float32))]

                    # Train Critic
                    if not self.one_opt:
                        train_value_ppo(self.value_net, mb_obs_seq, mb_returns_seq, self.optimizer_critic, lstm=True)

                    if not end_time == self.M:
                        for sind in reset_inds[ind]:
                            _ = self.reset_val_lstm_actor(reset_lstm_actor_states[sind])
                            _ = self.reset_val_lstm_critic(reset_lstm_critic_states[sind])
                    # Consider removing that one
                    if self.target_kl is not None:
                        if approx_kl > self.target_kl:
                            break

                # Check Divergence with Original Policy
                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

            # Build Episodic Return
            if len(ep_history) >= 1000:
                current_avg = sum(ep_history[-1000:]) / 1000
                avg_ep_history.append(current_avg)
                episodes.append(self.learned_episodes)

            elif ep_history:
                current_avg = sum(ep_history) / len(ep_history)
                avg_ep_history.append(current_avg)
                episodes.append(self.learned_episodes)

            # Print Some Learning Parameters
            if Verbose:
                print(f'UPDATE {update}:')
                print(f'Clip Fraction: {np.mean(clipfracs)} || KL Divergence: {approx_kl.numpy()} || Epochs Learned: {epoch} || Current Average Return: {current_avg}')

            if file_name:
                if lr_curve and len(avg_ep_history)%100==0:
                    plot_lr_curve(episodes, avg_ep_history, file_name)

                if not ((update + 1) % save_freq):
                    print("SAVING MODEL")
                    self.save(file_name)

    def save(self, file_name):
        save_optimizers(file_name, self.optimizer_actor, self.optimizer_critic)
        save_network_weights(file_name, self.value_net, self.action_net)

        info = {'input': self.envs.envs[0].type, 'input_dim': self.envs.observation_space.shape[0], 'number_of_Hidden_Layers_Actor': 2,
                'size_of_Hidden_Layers_Actor': 100, 'Gamma': self.gamma, 'lr_critic': self.lr_critic, 'lr_actor': self.lr_actor,
                'lr_critic_now': self.optimizer_critic.learning_rate.numpy(), 'lr_actor_now': self.optimizer_actor.learning_rate.numpy(),
                'Episode_Length': self.envs.envs[0].N_steps, 'Delta_t': self.envs.envs[0].dt, 'Env_Steps': self.envs.envs[0].env_steps,
                'Number_of_Episodes': self.learned_episodes, 'V_Swim': self.envs.envs[0].particle.vswim,
                'Target_dir': self.envs.envs[0].target_dir,
                'Number_of_Environments': self.N, 'Number_of_Steps_per_Env_per_Update': self.M, 'Number_of_Epochs_per_Update': self.epochs,
                'Number_of_Minibatches_per_Update': self.num_mini_batches, 'PPO_Clip_Coef': self.clip_coef, 'PPO_Entropy_Coef': self.ent_coef,
                'PPO_GAE_Lambda': self.gae_lambda, 'Learning_Algorithm_Name': self.name, 'Using_One_Optimizer': self.one_opt}

        save_info_general(file_name, info)

    def reset_ep_lstm_actor(self, ind):
        layers_states = []
        for layer in self.lstm_layers_actor:
            lstm_states = layer.states
            ht = lstm_states[0]
            ct = layer.states[1]
            reset_tensor = tf.transpose(tf.one_hot(ind * tf.ones((ht.shape[1],), dtype=tf.int32), ht.shape[0])) * -1 + 1
            layer.reset_states([tf.multiply(ht, reset_tensor), tf.multiply(ct, reset_tensor)])
            layers_states.append([tf.identity(layer.states[0]), tf.identity(layer.states[1])])

        return layers_states

    def reset_ep_lstm_critic(self, ind):
        layers_states = []
        for layer in self.lstm_layers_critic:
            lstm_states = layer.states
            ht = lstm_states[0]
            ct = layer.states[1]
            reset_tensor = tf.transpose(tf.one_hot(ind * tf.ones((ht.shape[1],), dtype=tf.int32), ht.shape[0])) * -1 + 1
            layer.reset_states([tf.multiply(ht, reset_tensor), tf.multiply(ct, reset_tensor)])
            layers_states.append([tf.identity(layer.states[0]), tf.identity(layer.states[1])])
        return layers_states

    def reset_val_lstm_actor(self, val):
        layers_states = []
        for ind, layer in enumerate(self.lstm_layers_actor):
            layer.reset_states([tf.identity(val[ind][0]), tf.identity(val[ind][1])])
            layers_states.append([tf.identity(val[ind][0]), tf.identity(val[ind][1])])

        return layers_states

    def reset_val_lstm_critic(self, val):
        layers_states = []
        for ind, layer in enumerate(self.lstm_layers_critic):
            layer.reset_states([tf.identity(val[ind][0]), tf.identity(val[ind][1])])
            layers_states.append([tf.identity(val[ind][0]), tf.identity(val[ind][1])])

        return layers_states

    def get_lstm_actor(self):
        return [[tf.identity(layer.states[0]), tf.identity(layer.states[1])] for layer in self.lstm_layers_actor]

    def get_lstm_critic(self):
        return [[tf.identity(layer.states[0]), tf.identity(layer.states[1])] for layer in self.lstm_layers_critic]


class Q_Learning:
    def __init__(self, env, learning_rate, states_per_variable, state_space_samples, epsilon, scaler, gamma,
                 learned_episodes=0, three_D=False):
        self.three_D = three_D
        self.learned_episodes = learned_episodes
        self.name = 'Q_learning'
        self.env = env
        self.learning_rate = learning_rate
        self.states_per_variable = states_per_variable
        self.epsilon = epsilon
        self.scaler = scaler
        self.gamma = gamma
        self.loss_critic = tf.keras.losses.MeanSquaredError()
        self.eval_steps = []
        self.eval_returns = []
        if self.three_D:
            self.num_actions = 6
        else:
            self.num_actions = 4

        self.intervals = [self.get_discrete_intervals_custom(scale_state(state_space_samples, self.scaler).numpy()[:, i])
                 for i in range(self.env.observation_space.shape[0])]

        # self.Q = np.zeros([self.states_per_variable ** env.observation_space.shape[0], self.num_actions])
        self.Q = np.ones([self.states_per_variable ** env.observation_space.shape[0], self.num_actions])*20.00

    def learn(self, num_episodes, file_name=False, save_freq=100000, Verbose=True, lr_curve=False):
        episode_history = []
        avg_ep_history = []
        for episode in range(num_episodes):
            state = self.get_discrete_state_custom(scale_state(self.env.reset(), self.scaler).numpy().reshape(-1))
            done = False
            while not done:
                frac = 1.0 - episode / (num_episodes - 1)
                self.epsilon = self.epsilon * frac
                if np.random.rand() <= self.epsilon:
                    action = np.random.choice(np.arange(self.num_actions))
                else:
                    action = np.random.choice(np.argwhere(self.Q[state, :] == np.amax(self.Q[state, :])).flatten())

                if self.three_D:
                    env_action = self.get_env_action3D(action)
                else:
                    env_action = self.get_env_action2D(action)
                next_env_state, reward, done, _ = self.env.step(env_action)
                next_state = self.get_discrete_state_custom(scale_state(next_env_state, self.scaler).numpy().reshape(-1))
                self.Q[state, action] = self.Q[state, action] + self.learning_rate * (
                            reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action])
                state = next_state

            episode_history.append(self.env.particle.z_distance_travelled(self.env.target_dir))
            self.learned_episodes += 1
            cond = False
            if Verbose:
                print("Episode: {}, Cumulative reward: {:0.2f}".format(episode, episode_history[-1]))

            if episode >= 999:
                current_avg = sum(episode_history[-1000:]) / 1000
                avg_ep_history.append(current_avg)

            else:
                current_avg = sum(episode_history) / len(episode_history)
                avg_ep_history.append(current_avg)

            if file_name:
                if not ((episode + 1) % save_freq):
                    print("SAVING MODEL")
                    self.save(file_name)

                if lr_curve and len(avg_ep_history)%100==0:
                    plot_lr_curve(np.arange(len(avg_ep_history)), avg_ep_history, file_name)
            if cond:
                break

    def save(self, file_name):
        np.save(file_name + '/Q_Matrix.npy', self.Q)
        np.save(file_name + '/Intervals.npy', np.array(self.intervals))

        info = {'Gamma': self.gamma, 'learning_rate': self.learning_rate,
                'Episode_Length': self.env.N_steps, 'Delta_t': self.env.dt,
                'Env_Steps': self.env.env_steps, 'Epsilon_now': self.epsilon,
                'States_per_Variable': self.states_per_variable, 'Number_of_Actions': self.num_actions,
                'Number_of_Episodes': self.learned_episodes, 'V_Swim': self.env.particle.vswim,
                'Target_dir': self.env.target_dir, 'Learning_Algorithm_Name': self.name}

        save_info_general(file_name, info)

    def get_discrete_intervals_custom(self, state_samples):
        probs = [(i + 1) / self.states_per_variable for i in range(self.states_per_variable - 1)]
        intervals = stats.mstats.mquantiles(state_samples, prob=probs)
        return intervals

    def get_discrete_state_custom(self, state):
        disc_state = np.zeros(state.shape)
        for ind, obs in enumerate(state):
            disc_state[ind] = bisect(self.intervals[ind], obs)

        disc_state = int(sum(disc_state * np.array([self.states_per_variable] * len(self.intervals)) ** np.arange(len(self.intervals))))
        return disc_state

    @staticmethod
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

    @staticmethod
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


def plot_lr_curve(steps, returns, file_name):
    fig, ax = Plot_Learning_Curve(steps, returns)
    fig.savefig(file_name + '/Real_Learning_Curve.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    np.save(file_name + '/lr_curve_steps.npy', np.array(steps))
    np.save(file_name + '/lr_curve_returns.npy', np.array(returns))
