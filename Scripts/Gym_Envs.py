import gym
from gym import spaces
from Scripts.Particles import *
from copy import deepcopy

class Env2D(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, Particle, Velocity_Field, Direction, time_step, N_steps, env_steps):
        super(Env2D, self).__init__()
        self.env_steps = env_steps
        self.min_action = -np.pi
        self.max_action = np.pi
        self.target_dir = Direction
        self.dt = time_step
        self.velf = Velocity_Field
        self.particle = deepcopy(Particle)
        self.n_steps = 0
        self.N_steps = N_steps
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=
        (1,), dtype=np.float32)

    def step(self, action):
        self.particle.theta = action[0]
        rew = 0.0
        for i in range(self.env_steps):
            pos_x, pos_y = self.particle.Next_Position(self.dt)
            rew += self.particle.last_z_distance(self.target_dir)

        self.state = self.pos_to_state(pos_x, pos_y)
        self.n_steps += 1
        done = False
        if self.n_steps >= self.N_steps:
            done = True

        return self.state, rew, done, {}

    def reset(self, scaler=False):
        self.n_steps = 0
        pos_x, pos_y = self.particle.reset()
        self.state  = self.pos_to_state(pos_x, pos_y)

        return self.state

    def pos_to_state(self, pos_x, pos_y):
        state = np.array([pos_x, pos_y], dtype=np.float32)

        return state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...

class VortEnv2D(Env2D):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)
        self.type = 'Vorticity'
        self.min_state = -1.0
        self.max_state = 1.0
        self.observation_space = spaces.Box(low=self.min_state, high=self.max_state, shape=
        (1,), dtype=np.float32)
        self.reset()

    def pos_to_state(self, pos_x, pos_y):
        if pos_x > self.velf.x[-1, -1]:
            pos_x -= self.velf.x[-1, -1]
        if pos_x < 0:
            pos_x += self.velf.x[-1, -1]
        if pos_y > self.velf.y[-1, -1]:
            pos_y -= self.velf.y[-1, -1]
        if pos_y < 0:
            pos_y += self.velf.y[-1, -1]

        obs = self.velf.Interpolate_Fields(pos_x, pos_y, self.velf.splrep_vortz)
        state = np.array([obs], dtype=np.float32)

        return state


class VelEnv2D(Env2D):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)
        self.type = 'Velocity'
        self.min_u = -0.5
        self.min_v = -0.5
        self.max_u = 0.5
        self.max_v = 0.5
        self.low_state = np.array(
            [self.min_u, self.min_v], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_u, self.max_v], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        self.reset()

    def pos_to_state(self, pos_x, pos_y):
        if pos_x > self.velf.x[-1, -1]:
            pos_x -= self.velf.x[-1, -1]
        if pos_x < 0:
            pos_x += self.velf.x[-1, -1]
        if pos_y > self.velf.y[-1, -1]:
            pos_y -= self.velf.y[-1, -1]
        if pos_y < 0:
            pos_y += self.velf.y[-1, -1]

        vx = self.velf.Interpolate_Fields(pos_x, pos_y, self.velf.splrep_vx)
        vy = self.velf.Interpolate_Fields(pos_x, pos_y, self.velf.splrep_vy)
        state = np.array([vx, vy])

        return state

class SurfEnv2D(Env2D):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)
        self.type = 'Surf'
        self.min_action = 0.0
        self.max_action = 10.0
        self.min_grad_ux = -0.5
        self.min_grad_uy = -0.5
        self.min_grad_vx = -0.5
        self.min_grad_vy = -0.5
        self.max_grad_ux = 0.5
        self.max_grad_uy = 0.5
        self.max_grad_vx = 0.5
        self.max_grad_vy = 0.5

        self.low_state = np.array(
            [self.min_grad_ux, self.min_grad_uy], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_grad_ux, self.max_grad_uy], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        self.reset()

    def pos_to_state(self, pos_x, pos_y):
        if pos_x > self.velf.x[-1, -1]:
            pos_x -= self.velf.x[-1, -1]
        if pos_x < 0:
            pos_x += self.velf.x[-1, -1]
        if pos_y > self.velf.y[-1, -1]:
            pos_y -= self.velf.y[-1, -1]
        if pos_y < 0:
            pos_y += self.velf.y[-1, -1]

        grad_array = self.velf.Obtain_Gradient(pos_x, pos_y)
        state = np.array([grad_array[0, 0], grad_array[0, 1]])

        return state

class GradEnv2D(Env2D):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)
        self.type = 'Gradients'
        self.min_grad_ux = -0.5
        self.min_grad_uy = -0.5
        self.max_grad_ux = 0.5
        self.max_grad_uy = 0.5

        self.low_state = np.array(
            [self.min_grad_ux, self.min_grad_uy], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_grad_ux, self.max_grad_uy], dtype=np.float32
        )

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        self.reset()

    def pos_to_state(self, pos_x, pos_y):
        if pos_x > self.velf.x[-1, -1]:
            pos_x -= self.velf.x[-1, -1]
        if pos_x < 0:
            pos_x += self.velf.x[-1, -1]
        if pos_y > self.velf.y[-1, -1]:
            pos_y -= self.velf.y[-1, -1]
        if pos_y < 0:
            pos_y += self.velf.y[-1, -1]

        grad_ux = self.velf.Interpolate_Fields(pos_x, pos_y, self.velf.splrep_gradvx_x)
        grad_uy = self.velf.Interpolate_Fields(pos_x, pos_y, self.velf.splrep_gradvx_y)
        state = np.array([grad_ux, grad_uy])

        return state

class FullGradEnv2D(Env2D):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)
        self.type = 'Gradients'
        self.min_grad_ux = -0.5
        self.min_grad_uy = -0.5
        self.min_grad_vx = -0.5
        self.max_grad_ux = 0.5
        self.max_grad_uy = 0.5
        self.max_grad_vx = 0.5

        self.low_state = np.array(
            [self.min_grad_ux, self.min_grad_uy,  self.min_grad_vx], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_grad_ux, self.max_grad_uy, self.max_grad_vx], dtype=np.float32
        )

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        self.reset()

    def pos_to_state(self, pos_x, pos_y):
        if pos_x > self.velf.x[-1, -1]:
            pos_x -= self.velf.x[-1, -1]
        if pos_x < 0:
            pos_x += self.velf.x[-1, -1]
        if pos_y > self.velf.y[-1, -1]:
            pos_y -= self.velf.y[-1, -1]
        if pos_y < 0:
            pos_y += self.velf.y[-1, -1]

        grad_ux = self.velf.Interpolate_Fields(pos_x, pos_y, self.velf.splrep_gradvx_x)
        grad_uy = self.velf.Interpolate_Fields(pos_x, pos_y, self.velf.splrep_gradvx_y)
        grad_vx = self.velf.Interpolate_Fields(pos_x, pos_y, self.velf.splrep_gradvy_x)
        state = np.array([grad_ux, grad_uy, grad_vx])

        return state

class VelVortEnv2D(Env2D):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)
        self.type = 'Vel_Vort'
        self.min_vort = -1.0
        self.min_u = -0.5
        self.min_v = -0.5
        self.max_vort = 1.0
        self.max_u = 0.5
        self.max_v = 0.5

        self.low_state = np.array(
            [self.min_u, self.min_v, self.min_vort], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_u, self.max_v, self.max_vort], dtype=np.float32
        )

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        self.reset()

    def pos_to_state(self, pos_x, pos_y):
        if pos_x > self.velf.x[-1, -1]:
            pos_x -= self.velf.x[-1, -1]
        if pos_x < 0:
            pos_x += self.velf.x[-1, -1]
        if pos_y > self.velf.y[-1, -1]:
            pos_y -= self.velf.y[-1, -1]
        if pos_y < 0:
            pos_y += self.velf.y[-1, -1]

        vx = self.velf.Interpolate_Fields(pos_x, pos_y, self.velf.splrep_vx)
        vy = self.velf.Interpolate_Fields(pos_x, pos_y, self.velf.splrep_vy)
        vort = self.velf.Interpolate_Fields(pos_x, pos_y, self.velf.splrep_vortz)
        state = np.array([vx, vy, vort])

        return state


class Env3D(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, Particle, Velocity_Field, Direction, time_step, N_steps, env_steps):
        super(Env3D, self).__init__()
        self.env_steps = env_steps
        self.min_x = -1.0
        self.max_x = 1.0
        self.min_y = -1.0
        self.max_y = 1.0
        self.min_z = -1.0
        self.max_z = 1.0
        self.target_dir = Direction
        self.dt = time_step
        self.n_steps = 0
        self.N_steps = N_steps
        self.velf = Velocity_Field
        self.particle = deepcopy(Particle)

        self.low_action = np.array(
            [self.min_x, self.min_y, self.min_z], dtype=np.float32
        )
        self.high_action = np.array(
            [self.max_x, self.max_y, self.max_z], dtype=np.float32
        )

        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)

    def step(self, action):
        self.particle.theta = np.arccos(action[2])
        self.particle.phi = np.arctan2(action[1], action[0])
        rew = 0.0
        for i in range(self.env_steps):
            pos_x, pos_y, pos_z = self.particle.Next_Position(self.dt)
            rew += self.particle.last_z_distance(self.target_dir)

        self.state = self.pos_to_state(pos_x, pos_y, pos_z)
        self.n_steps += 1
        done = False
        if self.n_steps >= self.N_steps:
            done = True

        return self.state, rew, done, {}

    def reset(self, scaler=False):
        self.n_steps = 0
        pos_x, pos_y, pos_z = self.particle.reset()
        self.state = self.pos_to_state(pos_x, pos_y, pos_z)

        return self.state

    def pos_to_state(self, pos_x, pos_y, pos_z):
        state = np.array([pos_x, pos_y, pos_z], dtype=np.float32)

        return state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...

class VelEnv3D(Env3D):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)
        self.type = 'Velocity'
        self.min_u = -Velocity_Field.A - Velocity_Field.C
        self.min_v = -Velocity_Field.B - Velocity_Field.A
        self.min_w = -Velocity_Field.C - Velocity_Field.B
        self.max_u = Velocity_Field.A + Velocity_Field.C
        self.max_v = Velocity_Field.B + Velocity_Field.A
        self.max_w = Velocity_Field.C + Velocity_Field.B

        self.low_state = np.array(
            [self.min_u, self.min_v, self.min_w], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_u, self.max_v, self.max_w], dtype=np.float32
        )

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        self.reset()

    def pos_to_state(self, pos_x, pos_y, pos_z):
        self.state = self.velf.Evaluate_ABC_Velocity(pos_x, pos_y, pos_z)

        return self.state


class VelGradEnv3D(Env3D):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)
        self.type = 'VelGrad'
        self.min_u = -Velocity_Field.A - Velocity_Field.C
        self.min_v = -Velocity_Field.B - Velocity_Field.A
        self.min_w = -Velocity_Field.C - Velocity_Field.B
        self.min_grad1 = -Velocity_Field.B - Velocity_Field.C
        self.min_grad2 = -Velocity_Field.A - Velocity_Field.B
        self.min_grad3 = -Velocity_Field.A - Velocity_Field.C
        self.max_u = Velocity_Field.A + Velocity_Field.C
        self.max_v = Velocity_Field.B + Velocity_Field.A
        self.max_w = Velocity_Field.C + Velocity_Field.B
        self.max_grad1 = Velocity_Field.B + Velocity_Field.C
        self.max_grad2 = Velocity_Field.A + Velocity_Field.B
        self.max_grad3 = Velocity_Field.A + Velocity_Field.C

        self.low_state = np.array(
            [self.min_u, self.min_v, self.min_w, self.min_grad1, self.min_grad2, self.min_grad3], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_u, self.max_v, self.max_w, self.max_grad1, self.max_grad2, self.max_grad3], dtype=np.float32
        )


        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        self.reset()

    def pos_to_state(self, pos_x, pos_y, pos_z):
        v = self.velf.Evaluate_ABC_Velocity(pos_x, pos_y, pos_z)
        gradv = self.velf.Evaluate_ABC_Gradient(pos_x, pos_y, pos_z)
        gradv1 = gradv[0, 1] + gradv[1, 0]
        gradv2 = gradv[0, 2] + gradv[2, 0]
        gradv3 = gradv[1, 2] + gradv[2, 1]
        gradv_sym = np.array([gradv1, gradv2, gradv3])
        self.state = np.concatenate((v, gradv_sym))

        return self.state

class VortEnv3D(Env3D):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)
        self.type = 'Vorticity'
        self.min_grad1 = -Velocity_Field.C - Velocity_Field.A
        self.min_grad2 = -Velocity_Field.A - Velocity_Field.B
        self.min_grad3 = -Velocity_Field.B - Velocity_Field.C
        self.max_grad1 = Velocity_Field.C + Velocity_Field.A
        self.max_grad2 = Velocity_Field.A + Velocity_Field.B
        self.max_grad3 = Velocity_Field.B + Velocity_Field.C

        self.low_state = np.array(
            [self.min_grad1, self.min_grad2, self.min_grad3], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_grad1, self.max_grad2, self.max_grad3], dtype=np.float32
        )


        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        self.reset()

    def pos_to_state(self, pos_x, pos_y, pos_z):
        gradv = self.velf.Evaluate_ABC_Gradient(pos_x, pos_y, pos_z)
        gradv1 = gradv[2, 1] - gradv[1, 2]
        gradv2 = gradv[0, 2] - gradv[2, 0]
        gradv3 = gradv[1, 0] - gradv[0, 1]
        self.state = np.array([gradv1, gradv2, gradv3])

        return self.state

class MagThetaEnv3D(Env3D):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)

        self.type = 'VelMag_Theta'
        self.min_vmag = 0.0
        self.max_vmag = np.sqrt(Velocity_Field.A ** 2 + Velocity_Field.B ** 2 + Velocity_Field.C ** 2 +
                                2 * Velocity_Field.A * np.sqrt(Velocity_Field.B ** 2 + Velocity_Field.C ** 2))
        self.min_angle = -1.0
        self.max_angle = 1.0


        self.low_state = np.array(
            [self.min_vmag, self.min_angle], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_vmag, self.max_angle], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        self.reset()


    def pos_to_state(self, pos_x, pos_y, pos_z):
        v = self.velf.Evaluate_ABC_Velocity(pos_x, pos_y, pos_z)
        v_mag = np.linalg.norm(v)
        v_norm = v / v_mag
        cosT = v_norm[0] * self.target_dir[0] + v_norm[1] * self.target_dir[1] + v_norm[2] * self.target_dir[2]
        self.state = np.array([v_mag, cosT])

        return self.state


class Env2DTran(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, Particle, Velocity_Field, Direction, time_step, N_steps, env_steps):
        super(Env2DTran, self).__init__()
        self.env_steps = env_steps

        self.min_action = -np.pi
        self.max_action = np.pi

        self.velf = Velocity_Field
        self.target_dir = Direction
        self.dt = time_step
        self.n_steps = 0
        self.N_steps = N_steps
        self.particle = deepcopy(Particle)


        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,),
                               dtype=np.float32)

    def step(self, action):
        self.particle.theta = action[0]
        rew = 0.0
        for i in range(self.env_steps):
            pos_x, pos_y = self.particle.Next_Position(self.dt)
            rew += self.particle.last_z_distance(self.target_dir)
        self.state = self.pos_to_state(pos_x, pos_y, self.particle.current_t)
        self.n_steps += 1
        done = False
        if self.n_steps >= self.N_steps:
            done = True

        return self.state, rew, done, {}

    def reset(self, scaler=False):
        self.n_steps = 0
        pos_x, pos_y = self.particle.reset(scaler=scaler)
        self.state = self.pos_to_state(pos_x, pos_y, self.particle.current_t)

        return self.state

    def pos_to_state(self, pos_x, pos_y, pos_t=None):
        state = np.array([pos_x, pos_y, pos_t], dtype=np.float32)

        return state


    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...

class GradEnvTran(Env2DTran):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)

        self.type = 'Gradients'
        self.min_grad_ux = -100.0
        self.min_grad_uy = -100.0
        self.min_grad_vx = -100.0

        self.max_grad_ux = 100.0
        self.max_grad_uy = 100.0
        self.max_grad_vx = 100.0


        self.low_state = np.array(
            [self.min_grad_ux, self.min_grad_uy, self.min_grad_vx], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_grad_ux, self.max_grad_uy, self.max_grad_vx], dtype=np.float32
        )


        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float64)
        self.reset()


    def pos_to_state(self, pos_x, pos_y, pos_t=None):
        if pos_x > self.velf.x_range:
            pos_x -= self.velf.x_range
        if pos_x < 0:
            pos_x += self.velf.x_range
        if pos_y > self.velf.y_range:
            pos_y -= self.velf.y_range
        if pos_y < 0:
            pos_y += self.velf.y_range

        if pos_t is None:
            grad = self.velf.Obtain_Gradient(pos_x, pos_y, self.particle.current_t)
        else:
            grad = self.velf.Obtain_Gradient(pos_x, pos_y, pos_t)
        grad_ux = grad[0, 0]
        grad_uy = grad[0, 1]
        grad_vx = grad[1, 0]
        state = np.array([grad_ux, grad_uy, grad_vx])

        return state


class ProjGradEnvTran(Env2DTran):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)

        self.type = 'Gradients'
        self.min_grad_ux = -100.0
        self.min_grad_uy = -100.0

        self.max_grad_ux = 100.0
        self.max_grad_uy = 100.0


        self.low_state = np.array(
            [self.min_grad_ux, self.min_grad_uy], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_grad_ux, self.max_grad_uy], dtype=np.float32
        )


        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float64)
        self.reset()


    def pos_to_state(self, pos_x, pos_y, pos_t=None):
        if pos_x > self.velf.x_range:
            pos_x -= self.velf.x_range
        if pos_x < 0:
            pos_x += self.velf.x_range
        if pos_y > self.velf.y_range:
            pos_y -= self.velf.y_range
        if pos_y < 0:
            pos_y += self.velf.y_range

        if pos_t is None:
            grad = self.velf.Obtain_Gradient(pos_x, pos_y, self.particle.current_t)
        else:
            grad = self.velf.Obtain_Gradient(pos_x, pos_y, pos_t)

        state = np.matmul(grad, [np.cos(self.particle.theta + np.pi/2), np.sin(self.particle.theta + np.pi/2)])

        return state

class RelGradEnvTran(Env2DTran):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)

        self.type = 'Gradients'
        self.min_grad_ux = -100.0
        self.min_grad_uy = -100.0
        self.min_grad_vx = -100.0

        self.max_grad_ux = 100.0
        self.max_grad_uy = 100.0
        self.max_grad_vx = 100.0


        self.low_state = np.array(
            [self.min_grad_ux, self.min_grad_uy, self.min_grad_vx], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_grad_ux, self.max_grad_uy, self.max_grad_vx], dtype=np.float32
        )


        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float64)
        self.reset()


    def pos_to_state(self, pos_x, pos_y, pos_t=None):
        if pos_x > self.velf.x_range:
            pos_x -= self.velf.x_range
        if pos_x < 0:
            pos_x += self.velf.x_range
        if pos_y > self.velf.y_range:
            pos_y -= self.velf.y_range
        if pos_y < 0:
            pos_y += self.velf.y_range

        if pos_t is None:
            grad = self.velf.Obtain_Gradient(pos_x, pos_y, self.particle.current_t)
        else:
            grad = self.velf.Obtain_Gradient(pos_x, pos_y, pos_t)

        R = np.array([[np.cos(self.particle.theta), -np.sin(self.particle.theta)],[np.sin(self.particle.theta), np.cos(self.particle.theta)]])
        rel_grad = np.linalg.multi_dot([R.T, grad, R])
        grad_ux = rel_grad[0, 0]
        grad_uy = rel_grad[0, 1]
        grad_vx = rel_grad[1, 0]
        state = np.array([grad_ux, grad_uy, grad_vx])

        return state

class FullGradEnvTran(Env2DTran):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)

        self.type = 'Gradients'
        self.min_grad_ux = -100.0
        self.min_grad_uy = -100.0
        self.min_grad_vx = -100.0
        self.min_grad_vy = -100.0

        self.max_grad_ux = 100.0
        self.max_grad_uy = 100.0
        self.max_grad_vx = 100.0
        self.max_grad_vy = 100.0


        self.low_state = np.array(
            [self.min_grad_ux, self.min_grad_uy, self.min_grad_vx, self.min_grad_vy], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_grad_ux, self.max_grad_uy, self.max_grad_vx, self.max_grad_vy], dtype=np.float32
        )


        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float64)
        self.reset()


    def pos_to_state(self, pos_x, pos_y, pos_t=None):
        if pos_x > self.velf.x_range:
            pos_x -= self.velf.x_range
        if pos_x < 0:
            pos_x += self.velf.x_range
        if pos_y > self.velf.y_range:
            pos_y -= self.velf.y_range
        if pos_y < 0:
            pos_y += self.velf.y_range

        if pos_t is None:
            grad = self.velf.Obtain_Gradient(pos_x, pos_y, self.particle.current_t)
        else:
            grad = self.velf.Obtain_Gradient(pos_x, pos_y, pos_t)
        grad_ux = grad[0, 0]
        grad_uy = grad[0, 1]
        grad_vx = grad[1, 0]
        grad_vy = grad[1, 1]
        state = np.array([grad_ux, grad_uy, grad_vx, grad_vy])

        return state


class VelEnvTran(Env2DTran):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)

        self.type = 'Velocity'
        self.min_u = -100.0
        self.min_v = -100.0
        self.max_u = 100.0
        self.max_v = 100.0


        self.low_state = np.array(
            [self.min_u, self.min_v], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_u, self.max_v], dtype=np.float32
        )


        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float64)
        self.reset()


    def pos_to_state(self, pos_x, pos_y, pos_t=None):
        if pos_x > self.velf.x_range:
            pos_x -= self.velf.x_range
        if pos_x < 0:
            pos_x += self.velf.x_range
        if pos_y > self.velf.y_range:
            pos_y -= self.velf.y_range
        if pos_y < 0:
            pos_y += self.velf.y_range

        if pos_t is None:
            grad = self.velf.Obtain_Velocity(pos_x, pos_y, self.particle.current_t)
        else:
            grad = self.velf.Obtain_Velocity(pos_x, pos_y, pos_t)

        return grad

class TauEnvTran(Env2DTran):
    def __init__(self, Particle=None, Velocity_Field=None, Direction=None, time_step=0.1, N_steps=10, env_steps=4):
        super().__init__(Particle, Velocity_Field, Direction, time_step, N_steps, env_steps)

        self.type = 'Tau'
        self.min_grad_ux = -100.0
        self.min_grad_uy = -100.0
        self.min_grad_vx = -100.0

        self.max_grad_ux = 100.0
        self.max_grad_uy = 100.0
        self.max_grad_vx = 100.0

        self.low_state = np.array(
            [self.min_grad_ux, self.min_grad_uy, self.min_grad_vx], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_grad_ux, self.max_grad_uy, self.max_grad_vx], dtype=np.float32
        )

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float64)
        self.reset()

    def step(self, action):
        rew = 0.0
        for i in range(self.env_steps):
            self.particle.theta = self.particle.surf(action[0], self.target_dir)
            pos_x, pos_y = self.particle.Next_Position(self.dt)
            rew += self.particle.last_z_distance(self.target_dir)
        self.state = self.pos_to_state(pos_x, pos_y, self.particle.current_t)
        self.n_steps += 1
        done = False
        if self.n_steps >= self.N_steps:
            done = True

        return self.state, rew, done, {}

    def pos_to_state(self, pos_x, pos_y, pos_t=None):
        if pos_x > self.velf.x_range:
            pos_x -= self.velf.x_range
        if pos_x < 0:
            pos_x += self.velf.x_range
        if pos_y > self.velf.y_range:
            pos_y -= self.velf.y_range
        if pos_y < 0:
            pos_y += self.velf.y_range

        if pos_t is None:
            grad = self.velf.Obtain_Gradient(pos_x, pos_y, self.particle.current_t)
        else:
            grad = self.velf.Obtain_Gradient(pos_x, pos_y, pos_t)
        grad_ux = grad[0, 0]
        grad_uy = grad[0, 1]
        grad_vx = grad[1, 0]
        state = np.array([grad_ux, grad_uy, grad_vx])

        return state
