import numpy as np
import scipy.linalg as sci
from copy import deepcopy


class PassiveParticle:
    def __init__(self, initial_x, initial_y, velocity_field):
        self.current_x = initial_x
        self.current_y = initial_y
        self.vel_field = velocity_field
        self.z_distance = 0
        self.history_x = [initial_x]
        self.history_y = [initial_y]
        self.history_dx = []
        self.history_dy = []

    def Next_Position(self, dt):
        dpos = self.RK_4(self.TGV_Model, [self.current_x, self.current_y], dt)
        dx = dpos[0]
        dy = dpos[1]
        self.history_dx.append(dx)
        self.history_dy.append(dy)
        x = self.current_x + dx
        y = self.current_y + dy
        if x > self.vel_field.x[-1, -1]:
            x -= self.vel_field.x[-1, -1] - self.vel_field.x[0, 0]
        if x < self.vel_field.x[0, 0]:
            x += self.vel_field.x[-1, -1] - self.vel_field.x[0, 0]
        if y > self.vel_field.y[-1, -1]:
            y -= self.vel_field.y[-1, -1] - self.vel_field.y[0, 0]
        if y < self.vel_field.y[0, 0]:
            y += self.vel_field.y[-1, -1] - self.vel_field.y[0, 0]

        self.current_x = x
        self.current_y = y
        self.history_x.append(x)
        self.history_y.append(y)

        return x, y


    def z_distance_travelled(self, z):
        unit_z = np.array(z) / np.linalg.norm(z)
        x_dist = sum(self.history_dx)
        y_dist = sum(self.history_dy)
        vect = np.array([x_dist, y_dist]).reshape((2,))
        z_dist = np.dot(vect, unit_z)

        return z_dist

    def last_z_distance(self, z):
        unit_z = np.array(z) / np.linalg.norm(z)
        vect = np.array([self.history_dx[-1], self.history_dy[-1]]).reshape((2,))
        z_dist = np.dot(vect, unit_z)

        return z_dist

    def Simulate_Trajectory(self, N, dt):
        for i in range(N):
            final_x, final_y = self.Next_Position(dt)

        return final_x, final_y


    def TGV_Model(self, x):
        u = -0.5 * np.cos(x[0]) * np.sin(x[1])
        v = 0.5 * np.sin(x[0]) * np.cos(x[1])
        xdot = np.array([u, v])

        return xdot

    def RK_4(self, f, x, dt):
        k1 = f(x)
        k2 = f(x + (dt / 2) * k1)
        k3 = f(x + (dt / 2) * k2)
        k4 = f(x + dt * k3)

        dx = (1 / 6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)

        return dx

    def get_absolute_positions(self):
        x = [self.history_x[0] + sum(self.history_dx[0:i]) for i, j in enumerate(self.history_dx)]
        y = [self.history_y[0] + sum(self.history_dy[0:i]) for i, j in enumerate(self.history_dy)]

        return x, y

    def reset(self):
        x = np.random.rand() * (self.vel_field.x[-1, -1] - self.vel_field.x[0, 0]) + self.vel_field.x[0, 0]
        y = np.random.rand() * (self.vel_field.y[-1, -1] - self.vel_field.y[0, 0]) + self.vel_field.y[0, 0]
        # x = 2.0
        # y = 3.0
        self.current_x = x
        self.current_y = y
        self.z_distance = 0
        self.history_x = [x]
        self.history_y = [y]
        self.history_dx = []
        self.history_dy = []

        return self.current_x, self.current_y

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        for k,v in self.__dict__.items():
            if k=='vel_field':
                setattr(result, k, self.vel_field)
            else:
                setattr(result, k, deepcopy(v, memo))

        return result


# This is an active particle that does not feel the vorticity of the flow
# the direction can be controlled directly

class ActiveParticle(PassiveParticle):
    def __init__(self, initial_x, initial_y, velocity_field, Theta, Vswim):
        super().__init__(initial_x, initial_y, velocity_field)
        self.theta = Theta
        self.vswim = Vswim

    def TGV_Model(self, x):
        u = -0.5 * np.cos(x[0]) * np.sin(x[1]) + self.vswim * np.cos(self.theta)
        v = 0.5 * np.sin(x[0]) * np.cos(x[1]) + self.vswim * np.sin(self.theta)
        xdot = np.array([u, v])

        return xdot

    def surf(self, tau, tar):
        gradient = self.vel_field.Obtain_Gradient(self.current_x, self.current_y)
        psurf = np.matmul(np.transpose(sci.expm(tau*gradient)), tar)
        psurf_norm = psurf/np.linalg.norm(psurf)
        Theta = np.arctan2(psurf_norm[1], psurf_norm[0])

        return Theta




class PassiveParticle3D:
    def __init__(self, initial_x, initial_y, initial_z, velocity_field):
        self.current_x = initial_x
        self.current_y = initial_y
        self.current_z = initial_z
        self.vel_field = velocity_field
        self.z_distance = 0
        self.history_x = [initial_x]
        self.history_y = [initial_y]
        self.history_z = [initial_z]
        self.history_dx = []
        self.history_dy = []
        self.history_dz = []

    def Next_Position(self, dt):
        dpos = self.RK_4(self.ABC_Model, [self.current_x, self.current_y, self.current_z], dt)
        dx = dpos[0]
        dy = dpos[1]
        dz = dpos[2]
        self.history_dx.append(dx)
        self.history_dy.append(dy)
        self.history_dz.append(dz)
        x = self.current_x + dx
        y = self.current_y + dy
        z = self.current_z + dz
        if x > self.vel_field.x[-1, -1, -1]:
            x -= self.vel_field.x[-1, -1, -1] - self.vel_field.x[0, 0, 0]
        if x < self.vel_field.x[0, 0, 0]:
            x += self.vel_field.x[-1, -1, -1] - self.vel_field.x[0, 0, 0]
        if y > self.vel_field.y[-1, -1, -1]:
            y -= self.vel_field.y[-1, -1, -1] - self.vel_field.y[0, 0, 0]
        if y < self.vel_field.y[0, 0, 0]:
            y += self.vel_field.y[-1, -1, -1] - self.vel_field.y[0, 0, 0]
        if z > self.vel_field.z[-1, -1, -1]:
            z -= self.vel_field.z[-1, -1, -1] - self.vel_field.z[0, 0, 0]
        if z < self.vel_field.z[0, 0, 0]:
            z += self.vel_field.z[-1, -1, -1] - self.vel_field.z[0, 0, 0]
        self.current_x = x
        self.current_y = y
        self.current_z = z
        self.history_x.append(x)
        self.history_y.append(y)
        self.history_z.append(z)

        return x, y, z

    def z_distance_travelled(self, z):
        unit_z = np.array(z) / np.linalg.norm(z)
        x_dist = sum(self.history_dx)
        y_dist = sum(self.history_dy)
        z_dist = sum(self.history_dz)
        vect = np.array([x_dist, y_dist, z_dist]).reshape((3,))
        z_dist = np.dot(vect, unit_z)

        return z_dist

    def last_z_distance(self, z):
        unit_z = np.array(z) / np.linalg.norm(z)
        vect = np.array([self.history_dx[-1], self.history_dy[-1], self.history_dz[-1]]).reshape((3,))
        z_dist = np.dot(vect, unit_z)

        return z_dist

    def Simulate_Trajectory(self, N, dt):
        for i in range(N):
            final_x, final_y, final_z = self.Next_Position(dt)

        return final_x, final_y, final_z

    def ABC_Model(self, x):
        xdot = self.vel_field.Evaluate_ABC_Velocity(x[0], x[1], x[2])

        return xdot

    def RK_4(self, f, x, dt):
        k1 = f(x)
        k2 = f(x + (dt / 2) * k1)
        k3 = f(x + (dt / 2) * k2)
        k4 = f(x + dt * k3)

        dx = (1 / 6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)

        return dx

    def get_absolute_positions(self):
        x = [self.history_x[0] + sum(self.history_dx[0:i]) for i, j in enumerate(self.history_dx)]
        y = [self.history_y[0] + sum(self.history_dy[0:i]) for i, j in enumerate(self.history_dy)]
        z = [self.history_z[0] + sum(self.history_dz[0:i]) for i, j in enumerate(self.history_dz)]

        return x, y, z

    def reset(self):
        x = np.random.rand()*(self.vel_field.x[-1, -1, -1] - self.vel_field.x[0, 0, 0]) + self.vel_field.x[0, 0, 0]
        y = np.random.rand()*(self.vel_field.y[-1, -1, -1] - self.vel_field.y[0, 0, 0]) + self.vel_field.y[0, 0, 0]
        z = np.random.rand()*(self.vel_field.z[-1, -1, -1] - self.vel_field.z[0, 0, 0]) + self.vel_field.z[0, 0, 0]
        self.current_x = x
        self.current_y = y
        self.current_z = z
        self.z_distance = 0
        self.history_x = [x]
        self.history_y = [y]
        self.history_z = [z]
        self.history_dx = []
        self.history_dy = []
        self.history_dz = []

        return self.current_x, self.current_y, self.current_z

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        for k,v in self.__dict__.items():
            if k=='vel_field':
                setattr(result, k, self.vel_field)
            else:
                setattr(result, k, deepcopy(v, memo))

        return result


class ActiveParticle3D(PassiveParticle3D):
    def __init__(self, initial_x, initial_y, initial_z, velocity_field, Theta, Phi, Vswim):
        super().__init__(initial_x, initial_y, initial_z, velocity_field)
        self.theta = Theta
        self.phi = Phi
        self.vswim = Vswim

    def ABC_Model(self, x):
        xdot = self.vel_field.Evaluate_ABC_Velocity(x[0], x[1], x[2])
        xdot[0] += self.vswim * np.sin(self.theta) * np.cos(self.phi)
        xdot[1] += self.vswim * np.sin(self.theta) * np.sin(self.phi)
        xdot[2] += self.vswim * np.cos(self.theta)

        return xdot

    def surf(self, tau, tar):
        gradient = self.vel_field.Evaluate_ABC_Gradient(self.current_x, self.current_y, self.current_z)
        psurf = np.matmul(np.transpose(sci.expm(tau*gradient)), tar)
        psurf_norm = psurf/np.linalg.norm(psurf)

        return psurf_norm

    def surf_vort(self, tau, tar):
        gradient = self.vel_field.Evaluate_ABC_Gradient(self.current_x, self.current_y, self.current_z)
        grad = 0.5*(gradient - gradient.T)
        psurf = np.matmul(np.transpose(sci.expm(tau*grad)), tar)
        psurf_norm = psurf/np.linalg.norm(psurf)

        return psurf_norm


class PassiveParticleTran:
    def __init__(self, initial_x, initial_y, initial_t, velocity_field):
        self.current_x = initial_x
        self.current_y = initial_y
        self.current_t = initial_t
        self.vel_field = velocity_field
        self.z_distance = 0
        self.history_x = [initial_x]
        self.history_y = [initial_y]
        self.history_t = [initial_t]
        self.history_dx = []
        self.history_dy = []

    def Next_Position(self, dt):
        dpos = self.RK_4(self.Model, [self.current_x, self.current_y], self.current_t, dt)
        dx = dpos[0]
        dy = dpos[1]
        self.history_dx.append(dx)
        self.history_dy.append(dy)
        x = self.current_x + dx
        y = self.current_y + dy
        t = self.current_t + dt
        if x > self.vel_field.x_range:
            x -= self.vel_field.x_range
        if x < 0:
            x += self.vel_field.x_range
        if y > self.vel_field.y_range:
            y -= self.vel_field.y_range
        if y < 0:
            y += self.vel_field.y_range
        self.current_x = x
        self.current_y = y
        self.current_t += dt
        self.history_x.append(x)
        self.history_y.append(y)
        self.history_t.append(t)

        return x, y

    def z_distance_travelled(self, z):
        unit_z = np.array(z) / np.linalg.norm(z)
        x_dist = sum(self.history_dx)
        y_dist = sum(self.history_dy)
        vect = np.array([x_dist, y_dist]).reshape((2,))
        z_dist = np.dot(vect, unit_z)

        return z_dist

    def last_z_distance(self, z):
        unit_z = np.array(z) / np.linalg.norm(z)
        vect = np.array([self.history_dx[-1], self.history_dy[-1]]).reshape((2,))
        z_dist = np.dot(vect, unit_z)

        return z_dist

    def Simulate_Trajectory(self, N, dt):
        for i in range(N):
            final_x, final_y = self.Next_Position(dt)

        return final_x, final_y

    def Model(self, x, t):
        if x[0] > self.vel_field.x_range:
            x[0] -= self.vel_field.x_range
        if x[0] < 0:
            x[0] += self.vel_field.x_range
        if x[1] > self.vel_field.y_range:
            x[1] -= self.vel_field.y_range
        if x[1] < 0:
            x[1] += self.vel_field.y_range

        u = self.vel_field.Interpolate_Fields(x[0], x[1], t, self.vel_field.vx)
        v = self.vel_field.Interpolate_Fields(x[0], x[1], t, self.vel_field.vy)
        xdot = np.array([float(u), float(v)])

        return xdot

    def RK_4(self, f, x, t, dt):
        k1 = f(x, t)
        k2 = f(x + (dt / 2) * k1, t + dt / 2)
        k3 = f(x + (dt / 2) * k2, t + dt / 2)
        k4 = f(x + dt * k3, t + dt)

        dx = (1 / 6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)

        return dx

    def get_absolute_positions(self):
        x = [self.history_x[0] + sum(self.history_dx[0:i]) for i, j in enumerate(self.history_dx)]
        y = [self.history_y[0] + sum(self.history_dy[0:i]) for i, j in enumerate(self.history_dy)]

        return x, y

    def reset(self, scaler=False):
        x = np.random.rand() * self.vel_field.x_range
        y = np.random.rand() * self.vel_field.y_range
        if scaler:
            t = np.random.rand()*self.vel_field.t_range
        else:
            t = np.random.rand()*(self.vel_field.t_range*3/4 - 5.0)
        self.current_x = x
        self.current_y = y
        self.current_t = t
        self.z_distance = 0
        self.history_x = [x]
        self.history_y = [y]
        self.history_t = [float(t)]
        self.history_dx = []
        self.history_dy = []

        return self.current_x, self.current_y

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        for k,v in self.__dict__.items():
            if k=='vel_field':
                setattr(result, k, self.vel_field)
            else:
                setattr(result, k, deepcopy(v, memo))

        return result


class ActiveParticleTran(PassiveParticleTran):
    def __init__(self, initial_x, initial_y, initial_t, velocity_field, Theta, Vswim, pedley=None):
        super().__init__(initial_x, initial_y, initial_t, velocity_field)
        self.theta = Theta
        self.vswim = Vswim
        if pedley == 0.0:
            self.pedley = None
        else:
            self.pedley = pedley
        self.current_px = np.cos(Theta)
        self.current_py = np.sin(Theta)
        self.history_px = [np.cos(Theta)]
        self.history_py = [np.sin(Theta)]

    def Model(self, x, t):
        if x[0] > self.vel_field.x_range:
            x[0] -= self.vel_field.x_range
        if x[0] < 0:
            x[0] += self.vel_field.x_range
        if x[1] > self.vel_field.y_range:
            x[1] -= self.vel_field.y_range
        if x[1] < 0:
            x[1] += self.vel_field.y_range

        u = self.vel_field.Interpolate_Fields(x[0], x[1], t, self.vel_field.vx) + self.vswim * np.cos(self.theta)
        v = self.vel_field.Interpolate_Fields(x[0], x[1], t, self.vel_field.vy) + self.vswim * np.sin(self.theta)

        xdot = np.array([float(u), float(v)])

        return xdot

    def Model_Pedley(self, x, t):
        if x[0] > self.vel_field.x_range:
            x[0] -= self.vel_field.x_range
        if x[0] < 0:
            x[0] += self.vel_field.x_range
        if x[1] > self.vel_field.y_range:
            x[1] -= self.vel_field.y_range
        if x[1] < 0:
            x[1] += self.vel_field.y_range

        tar = [np.cos(self.theta), np.sin(self.theta)]
        proj = np.dot(tar, [x[2], x[3]])
        vort = self.vel_field.Interpolate_Fields(x[0], x[1], t, self.vel_field.vort_z)
        dpdtx = -0.5 * vort * x[3] + (tar[0] - proj * x[2]) / (2 * self.pedley)
        dpdty = 0.5 * vort * x[2] + (tar[1] - proj * x[3]) / (2 * self.pedley)
        u = self.vel_field.Interpolate_Fields(x[0], x[1], t, self.vel_field.vx) + self.vswim * x[2]
        v = self.vel_field.Interpolate_Fields(x[0], x[1], t, self.vel_field.vy) + self.vswim * x[3]

        xdot = np.array([float(u), float(v), float(dpdtx), float(dpdty)])

        return xdot

    def Next_Position(self, dt):
        if self.pedley:
            dpos = self.RK_4(self.Model_Pedley, [self.current_x, self.current_y, self.current_px, self.current_py],
                             self.current_t, dt)
            px = self.current_px + dpos[2]
            py = self.current_py + dpos[3]
            norm_p = np.sqrt(px**2 + py**2)
            self.current_px = px/norm_p
            self.current_py = py/norm_p
            self.history_px.append(px/norm_p)
            self.history_py.append(py/norm_p)
        else:
            dpos = self.RK_4(self.Model, [self.current_x, self.current_y], self.current_t, dt)
        dx = dpos[0]
        dy = dpos[1]
        self.history_dx.append(dx)
        self.history_dy.append(dy)
        x = self.current_x + dx
        y = self.current_y + dy
        t = self.current_t + dt
        if x > self.vel_field.x_range:
            x -= self.vel_field.x_range
        if x < 0:
            x += self.vel_field.x_range
        if y > self.vel_field.y_range:
            y -= self.vel_field.y_range
        if y < 0:
            y += self.vel_field.y_range
        self.current_x = x
        self.current_y = y
        self.current_t += dt
        self.history_x.append(x)
        self.history_y.append(y)
        self.history_t.append(t)

        return x, y

    def surf(self, tau, tar):
        gradient = self.vel_field.Obtain_Gradient(self.current_x, self.current_y, self.current_t)
        psurf = np.matmul(np.transpose(sci.expm(tau * gradient)), tar)
        psurf_norm = psurf / np.linalg.norm(psurf)
        Theta = np.arctan2(psurf_norm[1], psurf_norm[0])

        return Theta

    def fit_symm_surf(self, alpha1, alpha2, alpha3, alpha4):
        gradient = self.vel_field.Obtain_Gradient(self.current_x, self.current_y, self.current_t)
        grad = gradient[1, :]
        vec = np.array([0.0, 1.0]) + np.array(
            [alpha1 * grad[0] + alpha2 * grad[1], alpha3 * grad[0] + alpha4 * grad[1]])
        vec_norm = vec / np.linalg.norm(vec)
        Theta = np.arctan2(vec_norm[1], vec_norm[0])

        return Theta

    def fit_symm_surf_squared(self, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8, alpha9, alpha10):
        gradient = self.vel_field.Obtain_Gradient(self.current_x, self.current_y, self.current_t)
        grad = gradient[1, :]
        vec = np.array([0.0, 1.0]) + np.array(
            [alpha1 * grad[0] + alpha2 * grad[1], alpha3 * grad[0] + alpha4 * grad[1]])
        vec += np.array([alpha5 * grad[0] ** 2 + alpha6 * grad[0] * grad[1] + alpha7 * grad[1] ** 2,
                         alpha8 * grad[0] ** 2 + alpha9 * grad[0] * grad[1] + alpha10 * grad[1] ** 2])
        vec_norm = vec / np.linalg.norm(vec)
        Theta = np.arctan2(vec_norm[1], vec_norm[0])

        return Theta

    def rel_surf(self, tau, tar):
        gradient = self.vel_field.Obtain_Gradient(self.current_x, self.current_y, self.current_t)
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                      [np.sin(self.theta), np.cos(self.theta)]])
        gradient = np.linalg.multi_dot([R.T, gradient, R])
        psurf = np.matmul(np.transpose(sci.expm(tau * gradient)), tar)
        psurf_norm = psurf / np.linalg.norm(psurf)
        Theta = np.arctan2(psurf_norm[1], psurf_norm[0])

        return Theta

    def reset(self, scaler=False):
        x = np.random.rand() * self.vel_field.x_range
        y = np.random.rand() * self.vel_field.y_range
        if scaler:
            t = np.random.rand() * self.vel_field.t_range
        else:
            t = np.random.rand() * (self.vel_field.t_range * 3 / 4 - 5.0)
        self.current_x = x
        self.current_y = y
        self.current_t = t
        self.z_distance = 0
        self.history_x = [x]
        self.history_y = [y]
        self.history_t = [float(t)]
        self.history_dx = []
        self.history_dy = []
        self.current_px = np.cos(np.pi / 2)
        self.current_py = np.sin(np.pi / 2)
        self.history_px = [np.cos(np.pi / 2)]
        self.history_py = [np.sin(np.pi / 2)]

        return self.current_x, self.current_y