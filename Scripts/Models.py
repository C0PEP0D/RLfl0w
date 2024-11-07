import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers


def value_function(input_dims):
    n_hidden1 = 100
    n_hidden2 = 100
    n_outputs = 1

    inputs = tf.keras.Input(shape=(input_dims,))

    dense_layer1 = layers.Dense(n_hidden1, activation='elu', kernel_initializer='glorot_uniform')
    x = dense_layer1(inputs)
    dense_layer2 = layers.Dense(n_hidden2, activation='elu', kernel_initializer='glorot_uniform')
    x = dense_layer2(x)
    kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005)
    prediction_layer = layers.Dense(n_outputs, kernel_initializer=kernel_initializer)
    prediction = prediction_layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=prediction)

    return model


def value_function_lstm(input_dims, num_envs):
    n_hidden1 = 100
    n_hidden2 = 100
    lstm_hidden = 60
    n_outputs = 1

    inputs = tf.keras.Input(shape=(None, input_dims), batch_size=num_envs)

    dense_layer1 = layers.Dense(n_hidden1, activation='elu', kernel_initializer='glorot_uniform')
    x = dense_layer1(inputs)
    # dense_layer2 = layers.Dense(n_hidden2, activation='elu', kernel_initializer='glorot_uniform')
    # x = dense_layer2(x)
    lstm_layer = layers.LSTM(lstm_hidden,
                             kernel_initializer='glorot_uniform',
                             recurrent_initializer='orthogonal',
                             bias_initializer='zeros',
                             return_sequences=True,
                             return_state=False,
                             stateful=True,
                             unroll=False)
    x = lstm_layer(x)
    kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005)
    prediction_layer = layers.Dense(n_outputs, kernel_initializer=kernel_initializer)
    prediction = prediction_layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=prediction)

    return model


def policy_network(input_dims, cyclic=True, conversion=tfp.distributions.Distribution.sample):
    n_hidden1 = 40
    n_hidden2 = 40
    n_outputs = 1

    inputs = tf.keras.Input(shape=(input_dims,))

    dense_layer1 = layers.Dense(n_hidden1, activation='elu', kernel_initializer='glorot_uniform')
    x = dense_layer1(inputs)
    dense_layer2 = layers.Dense(n_hidden2, activation='elu', kernel_initializer='glorot_uniform')
    x = dense_layer2(x)
    if cyclic:
        mu_layer = layers.Dense(n_outputs, kernel_initializer='glorot_uniform')
    else:
        bias_initializer = tf.keras.initializers.Constant(value=5.0)
        kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.005,
                                                                 maxval=0.005)  # be careful of this you might not need
        mu_layer = layers.Dense(n_outputs, kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer)
    mu = mu_layer(x)
    if cyclic:
        kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005)
        sigma_layer = layers.Dense(n_outputs, activation='relu', kernel_initializer=kernel_initializer)
    else:
        bias_initializer = tf.keras.initializers.Constant(value=5.0)
        sigma_layer0 = layers.Dense(n_outputs, kernel_initializer='glorot_uniform',
                                    bias_initializer=bias_initializer)
        x = sigma_layer0(x)
        sigma_layer = tf.keras.layers.Lambda(lambda y: tf.keras.backend.maximum(1e-8, y))
    sigma = sigma_layer(x)
    mu_sigma = [mu, sigma]
    if cyclic:
        dist_layer = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.VonMises(loc=t[0], concentration=t[1]),
            convert_to_tensor_fn=conversion)
    else:
        # dist_layer = tfp.layers.DistributionLambda(
        #     lambda t: tfp.distributions.Normal(t[0], t[1]), convert_to_tensor_fn=conversion)
        dist_layer = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.TruncatedNormal(t[0], t[1], 0.0, 1.0), convert_to_tensor_fn=conversion)

    dist = dist_layer(mu_sigma)
    model = tf.keras.Model(inputs=inputs, outputs=dist)

    return model


def policy_network_lstm(input_dims, num_envs, cyclic=True, conversion=tfp.distributions.Distribution.sample):
    n_hidden1 = 40
    n_hidden2 = 40
    lstm_hidden = 60
    n_outputs = 1

    inputs = tf.keras.Input(shape=(None, input_dims), batch_size=num_envs)

    dense_layer1 = layers.Dense(n_hidden1, activation='elu', kernel_initializer='glorot_uniform')
    x = dense_layer1(inputs)
    # dense_layer2 = layers.Dense(n_hidden2, activation='elu', kernel_initializer='glorot_uniform')
    # x = dense_layer2(x)
    lstm_layer = layers.LSTM(lstm_hidden,
                             kernel_initializer='glorot_uniform',
                             recurrent_initializer='orthogonal',
                             bias_initializer='zeros',
                             return_sequences=True,
                             return_state=False,
                             stateful=True,
                             unroll=False)
    x = lstm_layer(x)

    if cyclic:
        mu_layer = layers.Dense(n_outputs, kernel_initializer='glorot_uniform')
    else:
        bias_initializer = tf.keras.initializers.Constant(value=5.0)
        kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.005,
                                                                 maxval=0.005)  # be careful of this you might not need
        mu_layer = layers.Dense(n_outputs, kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer)
    mu = mu_layer(x)
    if cyclic:
        kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005)
        sigma_layer = layers.Dense(n_outputs, activation='relu', kernel_initializer=kernel_initializer)
    else:
        bias_initializer = tf.keras.initializers.Constant(value=5.0)
        sigma_layer0 = layers.Dense(n_outputs, kernel_initializer='glorot_uniform',
                                    bias_initializer=bias_initializer)
        x = sigma_layer0(x)
        sigma_layer = tf.keras.layers.Lambda(lambda y: tf.keras.backend.maximum(1e-8, y))
    sigma = sigma_layer(x)
    mu_sigma = [mu, sigma]
    if cyclic:
        dist_layer = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.VonMises(loc=t[0], concentration=t[1]),
            convert_to_tensor_fn=conversion)
    else:
        dist_layer = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(t[0], t[1]))
        # lambda t: tfp.distributions.TruncatedNormal(t[0], t[1], 0.0, 10.0))
    dist = dist_layer(mu_sigma)
    # dist = layers.TimeDistributed(dist_layer)(mu_sigma)
    model = tf.keras.Model(inputs=inputs, outputs=dist)

    return model


def policy_network3D(input_dims, conversion=tfp.distributions.Distribution.sample):
    n_hidden1 = 40
    n_hidden2 = 40
    n_outputs = 3

    inputs = tf.keras.Input(shape=(input_dims,))

    dense_layer1 = layers.Dense(n_hidden1, activation='elu', kernel_initializer='glorot_uniform')
    x = dense_layer1(inputs)
    dense_layer2 = layers.Dense(n_hidden2, activation='elu', kernel_initializer='glorot_uniform')
    x = dense_layer2(x)
    mu_layer = layers.Dense(n_outputs, kernel_initializer='glorot_uniform')
    mu = mu_layer(x)
    # unit_mu = tf.math.divide(mu, tf.math.sqrt(tf.reduce_sum(tf.math.square(mu))))
    unit_mu = tf.linalg.normalize(mu, ord='euclidean', axis=1, name=None)[0]
    kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005)
    sigma_layer = layers.Dense(1, activation='relu', kernel_initializer=kernel_initializer)
    sigma = tf.reshape(sigma_layer(x), [-1])
    # sigma = sigma_layer(x)
    mu_sigma = [unit_mu, sigma]
    dist_layer = tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.VonMisesFisher(mean_direction=t[0], concentration=t[1]),
        convert_to_tensor_fn=conversion)
    dist = dist_layer(mu_sigma)
    model = tf.keras.Model(inputs=inputs, outputs=dist)

    return model


def reset_ep_lstm_actor(actor, lstm_layers, ind):
    layers_states = []
    for layer in lstm_layers:
        lstm_states = layer.states
        ht = lstm_states[0]
        ct = layer.states[1]
        reset_tensor = tf.transpose(tf.one_hot(ind * tf.ones((ht.shape[1],), dtype=tf.int32), ht.shape[0])) * -1 + 1
        layer.reset_states([tf.multiply(ht, reset_tensor), tf.multiply(ct, reset_tensor)])
        layers_states.append([tf.identity(layer.states[0]), tf.identity(layer.states[1])])

    return layers_states


def reset_val_lstm_actor(actor, lstm_layers, val):
    layers_states = []
    for ind, layer in enumerate(lstm_layers):
        layer.reset_states([tf.identity(val[ind][0]), tf.identity(val[ind][1])])
        layers_states.append([tf.identity(val[ind][0]), tf.identity(val[ind][1])])

    return layers_states


def get_lstm_actor(lstm_layers):
    return [[tf.identity(layer.states[0]), tf.identity(layer.states[1])] for layer in lstm_layers]


def loss_actor(alpha):
    def loss(y_true, y_pred):
        return tf.matmul(-tf.reshape(y_pred.log_prob(y_true), [-1, 1]), alpha)
        # return tf.matmul(-y_pred.log_prob(y_true), alpha)

    return loss


@tf.function
def train_value_net(model, state, target, optimizer, loss):
    with tf.GradientTape() as tape:
        V_of_state = model(state)
        lossv = loss(target, V_of_state)

    grads = tape.gradient(lossv, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return V_of_state


@tf.function
def train_value_net_2(model, state, next_state, reward, optimizer, loss):
    with tf.GradientTape() as tape:
        V_of_state = model(state)
        V_of_next_state = model(next_state)
        target = tf.math.add(V_of_next_state, reward)
        lossv = loss(target, V_of_state)

    grads = tape.gradient(lossv, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return V_of_state, target


@tf.function
def train_action_net(model, state, alpha, action, optimizer):
    with tf.GradientTape() as tape:
        actor_dist = model(state)
        lossa = loss_actor(alpha)(action, actor_dist)

    gradsa = tape.gradient(lossa, model.trainable_variables)
    optimizer.apply_gradients(zip(gradsa, model.trainable_variables))


@tf.function
def gradients_value_net(model, state, target, loss):
    with tf.GradientTape() as tape:
        V_of_state = model(state)
        lossv = loss(target, V_of_state)

    grads = tape.gradient(lossv, model.trainable_variables)
    return V_of_state, grads


@tf.function
def gradients_action_net(model, state, alpha, action):
    with tf.GradientTape() as tape:
        actor_dist = model(state)
        lossa = loss_actor(alpha)(action, actor_dist)

    gradsa = tape.gradient(lossa, model.trainable_variables)
    return gradsa


def get_sample_and_logprob(dist):
    x = dist.sample()
    y = tf.reshape(dist.log_prob(x), [-1, 1])

    return tf.concat([x, y], 1)


def get_sample_and_logprob_lstm(dist):
    x = dist.sample()
    y = dist.log_prob(x)

    return tf.concat([x, y], -1)


def get_mean(dist):
    x = dist.loc
    return x


def get_mean3D(dist):
    x = dist.mean_direction
    return x


@tf.function
def train_action_ppo(action_net, mb_obs, clip_coef, ent_coef, mb_actions, mb_advantages, mb_logprobs, optimizer,
                     lstm=False):
    with tf.GradientTape() as tape:
        new_log_action = action_net(mb_obs)
        if lstm:
            new_log_probs = tf.squeeze(new_log_action.log_prob(tf.cast(mb_actions, dtype=tf.float32)), axis=-1)
            entropy = tf.squeeze(new_log_action.entropy(), axis=-1)
        else:
            new_log_probs = tf.reshape(new_log_action.log_prob(tf.cast(mb_actions, dtype=tf.float32)), [-1])
            entropy = tf.reshape(new_log_action.entropy(), [-1])
        logratio = new_log_probs - tf.cast(mb_logprobs, dtype=tf.float32)
        ratio = tf.math.exp(logratio)

        # Remove the Mean from the Advantages
        mb_advantages = (mb_advantages - tf.math.reduce_mean(mb_advantages)) / (
                tf.math.reduce_std(mb_advantages) + tf.constant(1e-8))

        # Calculate actor loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * tf.clip_by_value(ratio, tf.constant(1.0) - clip_coef, tf.constant(1.0) + clip_coef)
        pg_loss = tf.math.reduce_mean(tf.math.maximum(pg_loss1, pg_loss2))

        # Entropy loss
        ent_loss = -tf.math.reduce_mean(entropy) * ent_coef

        # Final loss
        pg_ent_loss = pg_loss + ent_loss

    gradsa = tape.gradient(pg_ent_loss, action_net.trainable_variables)
    optimizer.apply_gradients(zip(gradsa, action_net.trainable_variables))

    return logratio, ratio


@tf.function
def train_action_ppo_noent(action_net, mb_obs, clip_coef, mb_actions, mb_advantages, mb_logprobs, optimizer,
                           lstm=False):
    # DO THIS FOR SYMMETRIC OUTPUT:
    # mb_obs = tf.concat([mb_obs, tf.stack([mb_obs[:, 0], -mb_obs[:, 1], -mb_obs[:, 2]], axis=1)], axis=0)
    # mb_actions = tf.concat([mb_actions, tf.experimental.numpy.pi - mb_actions], axis=0)
    # mb_advantages = tf.concat([mb_advantages, mb_advantages], axis=0)
    # mb_logprobs = tf.concat([mb_logprobs, mb_logprobs], axis=0)
    #######
    with tf.GradientTape() as tape:
        new_log_action = action_net(mb_obs)
        if lstm:
            new_log_probs = tf.squeeze(new_log_action.log_prob(tf.cast(mb_actions, dtype=tf.float32)), axis=-1)
        else:
            new_log_probs = tf.reshape(new_log_action.log_prob(tf.cast(mb_actions, dtype=tf.float32)), [-1])
        logratio = new_log_probs - tf.cast(mb_logprobs, dtype=tf.float32)
        ratio = tf.math.exp(logratio)

        # Remove the Mean from the Advantages
        mb_advantages = (mb_advantages - tf.math.reduce_mean(mb_advantages)) / (
                tf.math.reduce_std(mb_advantages) + tf.constant(1e-8))

        # Calculate actor loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * tf.clip_by_value(ratio, tf.constant(1.0) - clip_coef, tf.constant(1.0) + clip_coef)
        pg_loss = tf.math.reduce_mean(tf.math.maximum(pg_loss1, pg_loss2))

    gradsa = tape.gradient(pg_loss, action_net.trainable_variables)
    optimizer.apply_gradients(zip(gradsa, action_net.trainable_variables))

    return logratio, ratio


@tf.function
def train_value_ppo(value_net, mb_obs, mb_returns, optimizer, lstm=False):
    with tf.GradientTape() as tape:
        if lstm:
            new_values = tf.squeeze(value_net(mb_obs), axis=-1)
        else:
            new_values = tf.reshape(value_net(mb_obs), [-1])
        v_loss = tf.math.reduce_mean(0.5 * ((new_values - mb_returns) ** 2))

    gradsa = tape.gradient(v_loss, value_net.trainable_variables)
    optimizer.apply_gradients(zip(gradsa, value_net.trainable_variables))


@tf.function
def train_everything_ppo(action_net, value_net, mb_obs, clip_coef, ent_coef, mb_actions, mb_advantages,
                         mb_returns, mb_logprobs, optimizer, lstm=False):
    with tf.GradientTape() as tape:
        new_log_action = action_net(mb_obs)
        if lstm:
            new_log_probs = tf.squeeze(new_log_action.log_prob(tf.cast(mb_actions, dtype=tf.float32)), axis=-1)
            entropy = tf.squeeze(new_log_action.entropy(), axis=-1)
        else:
            new_log_probs = tf.reshape(new_log_action.log_prob(tf.cast(mb_actions, dtype=tf.float32)), [-1])
            entropy = tf.reshape(new_log_action.entropy(), [-1])
        logratio = new_log_probs - tf.cast(mb_logprobs, dtype=tf.float32)
        ratio = tf.math.exp(logratio)

        # Remove the Mean from the Advantages
        mb_advantages = (mb_advantages - tf.math.reduce_mean(mb_advantages)) / (
                tf.math.reduce_std(mb_advantages) + tf.constant(1e-8))

        # Value Loss
        if lstm:
            new_values = tf.squeeze(value_net(mb_obs), axis=-1)
        else:
            new_values = tf.reshape(value_net(mb_obs), [-1])
        v_loss = tf.math.reduce_mean(tf.constant(0.5) * ((new_values - mb_returns) ** tf.constant(2.0)))

        # Calculate actor loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * tf.clip_by_value(ratio, tf.constant(1.0) - clip_coef, tf.constant(1.0) + clip_coef)
        pg_loss = tf.math.reduce_mean(tf.math.maximum(pg_loss1, pg_loss2))

        # Entropy loss
        ent_loss = -tf.math.reduce_mean(entropy) * ent_coef

        # Final loss
        loss = pg_loss + ent_loss + v_loss

    grads = tape.gradient(loss, action_net.trainable_variables + value_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, action_net.trainable_variables + value_net.trainable_variables))

    return logratio, ratio


@tf.function
def train_everything_ppo_noent(action_net, value_net, mb_obs, clip_coef, mb_actions, mb_advantages,
                               mb_returns, mb_logprobs, optimizer, lstm=False):
    with tf.GradientTape() as tape:
        new_log_action = action_net(mb_obs)
        if lstm:
            new_log_probs = tf.squeeze(new_log_action.log_prob(tf.cast(mb_actions, dtype=tf.float32)), axis=-1)
        else:
            new_log_probs = tf.reshape(new_log_action.log_prob(tf.cast(mb_actions, dtype=tf.float32)), [-1])
        logratio = new_log_probs - tf.cast(mb_logprobs, dtype=tf.float32)
        ratio = tf.math.exp(logratio)

        # Remove the Mean from the Advantages
        mb_advantages = (mb_advantages - tf.math.reduce_mean(mb_advantages)) / (
                tf.math.reduce_std(mb_advantages) + tf.constant(1e-8))

        # Value Loss
        if lstm:
            new_values = tf.squeeze(value_net(mb_obs), axis=-1)
        else:
            new_values = tf.reshape(value_net(mb_obs), [-1])
        v_loss = tf.math.reduce_mean(tf.constant(0.5) * ((new_values - mb_returns) ** tf.constant(2.0)))

        # Calculate actor loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * tf.clip_by_value(ratio, tf.constant(1.0) - clip_coef, tf.constant(1.0) + clip_coef)
        pg_loss = tf.math.reduce_mean(tf.math.maximum(pg_loss1, pg_loss2))

        # Final loss
        loss = pg_loss + v_loss

    grads = tape.gradient(loss, action_net.trainable_variables + value_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, action_net.trainable_variables + value_net.trainable_variables))

    return logratio, ratio, grads
