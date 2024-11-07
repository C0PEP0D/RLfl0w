import numpy as np
import math
from scipy import interpolate


class VelField:

    def __init__(self, x_range, y_range, x_density, y_density):
        self.x, self.y = np.meshgrid(np.linspace(0, x_range, x_density), np.linspace(0, y_range, y_density))
        self.vx = np.zeros(self.x.shape)
        self.vy = np.zeros(self.y.shape)
        self.grad_vx = np.zeros(self.x.shape + (2,))
        self.grad_vy = np.zeros(self.y.shape + (2,))
        self.vort_z = np.zeros(self.x.shape)

    def Generate_TGV(self):
        self.vx = -0.5 * np.cos(self.x) * np.sin(self.y)
        self.vy = 0.5 * np.sin(self.x) * np.cos(self.y)
        self.vort_z = np.cos(self.x) * np.cos(self.y)
        self.grad_vx[:, :, 0] = 0.5 * np.sin(self.x) * np.sin(self.y)
        self.grad_vx[:, :, 1] = -0.5 * np.cos(self.x) * np.cos(self.y)
        self.grad_vy[:, :, 0] = 0.5 * np.cos(self.x) * np.cos(self.y)
        self.grad_vy[:, :, 1] = -0.5 * np.sin(self.x) * np.sin(self.y)

        self.splrep_vx = interpolate.bisplrep(self.x, self.y, self.vx, s=0)
        self.splrep_vy = interpolate.bisplrep(self.x, self.y, self.vy, s=0)
        self.splrep_gradvx_x = interpolate.bisplrep(self.x, self.y, self.grad_vx[:, :, 0], s=0)
        self.splrep_gradvx_y = interpolate.bisplrep(self.x, self.y, self.grad_vx[:, :, 1], s=0)
        self.splrep_gradvy_x = interpolate.bisplrep(self.x, self.y, self.grad_vy[:, :, 0], s=0)
        self.splrep_gradvy_y = interpolate.bisplrep(self.x, self.y, self.grad_vy[:, :, 1], s=0)
        self.splrep_vortz = interpolate.bisplrep(self.x, self.y, self.vort_z, s=0)

    def Interpolate_Fields(self, x_eval, y_eval, tck):
        int_values = interpolate.bisplev(x_eval, y_eval, tck)
        return int_values

    def Obtain_Gradient(self, x_eval, y_eval):
        gradient_array = np.zeros((2, 2))

        gradient_array[0, 0] = self.Interpolate_Fields(x_eval, y_eval, self.splrep_gradvx_x)
        gradient_array[0, 1] = self.Interpolate_Fields(x_eval, y_eval, self.splrep_gradvx_y)
        gradient_array[1, 0] = self.Interpolate_Fields(x_eval, y_eval, self.splrep_gradvy_x)
        gradient_array[1, 1] = self.Interpolate_Fields(x_eval, y_eval, self.splrep_gradvy_y)

        return gradient_array

    # def Plot_Vel_Field(self):
    #     plt.figure()
    #     plt.quiver(self.x, self.y, self.vx, self.vy)
    #     plt.title("Velocity Field.")
    #     plt.show()
    #
    # def Plot_Vorticity(self):
    #     plt.figure()
    #     plt.pcolormesh(self.x, self.y, self.vort_z, cmap='RdBu', shading='gouraud')
    #     plt.colorbar()
    #     plt.title("Vorticity.")
    #     plt.show()
    #
    # def Plot_Gradients(self):
    #     fig, axs = plt.subplots(2, 2)
    #     axs[0, 0].pcolormesh(self.x, self.y, self.grad_vx[:, :, 0], cmap='RdBu', shading='gouraud')
    #     axs[0, 0].set_title('Grad Ux')
    #     axs[0, 1].pcolormesh(self.x, self.y, self.grad_vx[:, :, 1], cmap='RdBu', shading='gouraud')
    #     axs[0, 1].set_title('Grad Uy')
    #     axs[1, 0].pcolormesh(self.x, self.y, self.grad_vy[:, :, 0], cmap='RdBu', shading='gouraud')
    #     axs[1, 0].set_title('Grad Vx')
    #     axs[1, 1].pcolormesh(self.x, self.y, self.grad_vy[:, :, 1], cmap='RdBu', shading='gouraud')
    #     axs[1, 1].set_title('Grad Vy')

        # for ax in axs.flat:
        #     ax.set(xlabel='x', ylabel='y')
        #
        # # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axs.flat:
        #     ax.label_outer()


class VelField3D():

    def __init__(self, x_range, y_range, z_range, x_density, y_density, z_density):
        self.x, self.y, self.z = np.meshgrid(np.linspace(0, x_range, x_density), np.linspace(0, y_range, y_density),
                                             np.linspace(0, z_range, z_density))
        self.grad_vx = np.zeros(self.x.shape + (3,))
        self.grad_vy = np.zeros(self.y.shape + (3,))
        self.grad_vz = np.zeros(self.z.shape + (3,))

    def Generate_ABC(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.vx = A * np.sin(self.z) + C * np.cos(self.y)
        self.vy = B * np.sin(self.x) + A * np.cos(self.z)
        self.vz = C * np.sin(self.y) + B * np.cos(self.x)
        self.grad_vx[:, :, :, 0] = np.zeros(self.x.shape)
        self.grad_vx[:, :, :, 1] = -C * np.sin(self.y)
        self.grad_vx[:, :, :, 2] = A * np.cos(self.z)
        self.grad_vy[:, :, :, 0] = B * np.cos(self.x)
        self.grad_vy[:, :, :, 1] = np.zeros(self.x.shape)
        self.grad_vy[:, :, :, 2] = -A * np.sin(self.z)
        self.grad_vz[:, :, :, 0] = -B * np.sin(self.x)
        self.grad_vz[:, :, :, 1] = C * np.cos(self.y)
        self.grad_vz[:, :, :, 2] = np.zeros(self.x.shape)

    def Evaluate_ABC_Velocity(self, x_eval, y_eval, z_eval):
        v = np.zeros(3, )
        v[0] = self.A * np.sin(z_eval) + self.C * np.cos(y_eval)
        v[1] = self.B * np.sin(x_eval) + self.A * np.cos(z_eval)
        v[2] = self.C * np.sin(y_eval) + self.B * np.cos(x_eval)

        return v

    def Evaluate_ABC_Gradient(self, x_eval, y_eval, z_eval):
        grad_v = np.zeros((3, 3))
        grad_v[0, 0] = np.float64(0)
        grad_v[0, 1] = -self.C * np.sin(y_eval)
        grad_v[0, 2] = self.A * np.cos(z_eval)
        grad_v[1, 0] = self.B * np.cos(x_eval)
        grad_v[1, 1] = np.float64(0)
        grad_v[1, 2] = -self.A * np.sin(z_eval)
        grad_v[2, 0] = -self.B * np.sin(x_eval)
        grad_v[2, 1] = self.C * np.cos(y_eval)
        grad_v[2, 2] = np.float64(0)

        return grad_v

    # def Plot_Vel_Field(self):
    #     ax = plt.subplot(projection='3d')
    #     ax.quiver(self.x[::5, ::5, ::5], self.y[::5, ::5, ::5], self.z[::5, ::5, ::5], self.vx[::5, ::5, ::5],
    #               self.vy[::5, ::5, ::5], self.vz[::5, ::5, ::5], length=0.1)
    #     ax.set_title("Velocity Field.")
    #
    # def Plot_Velocity_Magnitude(self):
    #     v_mag = np.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
    #     ax = plt.subplot(projection='3d')
    #     img = ax.scatter(self.x[::5, ::5, ::5], self.y[::5, ::5, ::5], self.z[::5, ::5, ::5],
    #                      c=v_mag[::5, ::5, ::5], s=100, cmap="Spectral")
    #     plt.colorbar(img)
    #
    #     # adding title and labels
    #     ax.set_title("3D Velocity Magnitude")
    #     ax.set_xlabel('X-axis')
    #     ax.set_ylabel('Y-axis')
    #     ax.set_zlabel('Z-axis')


class VelFieldTran:
    def __init__(self, x_range, y_range, t_range, x_density, y_density, t_density, vx, vy):
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range
        self.dx = self.x_range / x_density
        self.dy = self.y_range / y_density
        self.x_density = x_density
        self.y_density = y_density
        self.t_density = t_density
        self.x_space = np.linspace(self.dx / 2, x_range - self.dx / 2, x_density)
        self.y_space = np.linspace(self.dy / 2, y_range - self.dy / 2, y_density)
        self.t_space = np.linspace(0, t_range, t_density)
        self.x, self.y = np.meshgrid(self.x_space, self.y_space)
        self.vx = vx
        self.vy = vy

        g = np.concatenate((self.vx[1:, :, :], self.vx[0:1, :, :]), axis=0)
        f = np.concatenate((self.vx[-1, :, :].reshape((1, self.y_density, self.t_density)), self.vx[:-1, :, :]), axis=0)
        e = np.concatenate((self.vx[:, 1:, :], self.vx[:, 0:1, :]), axis=1)
        k = np.concatenate((self.vx[:, -1, :].reshape((256, 1, 1000)), self.vx[:, :-1, :]), axis=1)

        c = np.concatenate((self.vy[1:, :, :], self.vy[0:1, :, :]), axis=0)
        m = np.concatenate((self.vy[-1, :, :].reshape((1, self.y_density, self.t_density)), self.vy[:-1, :, :]), axis=0)
        b = np.concatenate((self.vy[:, 1:, :], self.vy[:, 0:1, :]), axis=1)
        l = np.concatenate((self.vy[:, -1, :].reshape((256, 1, 1000)), self.vy[:, :-1, :]), axis=1)

        self.dudx = (g - f) / (2 * self.dx)
        self.dudy = (e - k) / (2 * self.dy)
        self.dvdx = (c - m) / (2 * self.dx)
        self.dvdy = (b - l) / (2 * self.dy)
        self.vort_z = self.dvdx - self.dudy

    def Interpolate_Fields(self, x_eval, y_eval, t_eval, field):
        int_values = self.Trilinear_Interp(field, x_eval, y_eval, t_eval)
        return int_values

    def Trilinear_Interp(self, u, xcoord, ycoord, tcoord):
        t = tcoord / self.t_range * (self.t_density - 1)
        ut0 = np.zeros((2, 2))
        ut1 = np.zeros((2, 2))
        t0 = math.floor(t) % self.t_density
        t1 = (t0 + 1)

        if (xcoord < self.dx / 2 and ycoord < self.dy / 2) or (
                xcoord >= self.x_range - self.dx / 2 and ycoord < self.dy / 2) or (
                xcoord < self.dx / 2 and ycoord >= self.y_range - self.dy / 2) or (
                xcoord >= self.x_range - self.dx / 2 and ycoord >= self.y_range - self.dy / 2):
            if xcoord < self.dx / 2:
                x = xcoord / self.dx + 0.5
                x0 = 0.0
            else:
                x = (xcoord - (self.x_range - self.dx / 2)) / self.dx
                x0 = 0.0

            if ycoord < self.dy / 2:
                y = ycoord / self.dy + 0.5
                y0 = 0.0
            else:
                y = (ycoord - (self.y_range - self.dy / 2)) / self.dy
                y0 = 0.0

            ut0[0, 0] = u[-1, -1, t0]
            ut0[1, 0] = u[0, -1, t0]
            ut0[0, 1] = u[-1, 0, t0]
            ut0[1, 1] = u[0, 0, t0]

            ut1[0, 0] = u[-1, -1, t1]
            ut1[1, 0] = u[0, -1, t1]
            ut1[0, 1] = u[-1, 0, t1]
            ut1[1, 1] = u[0, 0, t1]

        elif xcoord < self.dx / 2 or xcoord >= self.x_range - self.dx / 2:
            if xcoord < self.dx / 2:
                x = xcoord / self.dx + 0.5
                x0 = 0.0
            else:
                x = (xcoord - (self.x_range - self.dx / 2)) / self.dx
                x0 = 0.0
            y = (ycoord - self.dy / 2) / (self.y_range) * (self.y_density - 1)
            y0 = math.floor(y) % self.y_density
            y1 = (y0 + 1)
            ut0[0, 0] = u[-1, y0, t0]
            ut0[1, 0] = u[0, y0, t0]
            ut0[0, 1] = u[-1, y1, t0]
            ut0[1, 1] = u[0, y1, t0]

            ut1[0, 0] = u[-1, y0, t1]
            ut1[1, 0] = u[0, y0, t1]
            ut1[0, 1] = u[-1, y1, t1]
            ut1[1, 1] = u[0, y1, t1]

        elif ycoord < self.dy / 2 or ycoord >= self.y_range - self.dy / 2:
            if ycoord < self.dy / 2:
                y = ycoord / self.dy + 0.5
                y0 = 0.0
            else:
                y = (ycoord - (self.y_range - self.dy / 2)) / self.dy
                y0 = 0.0
            x = (xcoord - self.dx / 2) / (self.x_range) * (self.x_density - 1)
            x0 = math.floor(x) % self.x_density
            x1 = (x0 + 1)
            ut0[0, 0] = u[x0, -1, t0]
            ut0[1, 0] = u[x1, -1, t0]
            ut0[0, 1] = u[x0, 0, t0]
            ut0[1, 1] = u[x1, 0, t0]

            ut1[0, 0] = u[x0, -1, t1]
            ut1[1, 0] = u[x1, -1, t1]
            ut1[0, 1] = u[x0, 0, t1]
            ut1[1, 1] = u[x1, 0, t1]

        else:
            x = (xcoord - self.dx / 2) / (self.x_range) * (self.x_density - 1)
            y = (ycoord - self.dy / 2) / (self.y_range) * (self.y_density - 1)
            x0 = math.floor(x) % self.x_density
            y0 = math.floor(y) % self.y_density
            x1 = (x0 + 1)
            y1 = (y0 + 1)
            ut0[0, 0] = u[x0, y0, t0]
            ut0[1, 0] = u[x1, y0, t0]
            ut0[0, 1] = u[x0, y1, t0]
            ut0[1, 1] = u[x1, y1, t0]

            ut1[0, 0] = u[x0, y0, t1]
            ut1[1, 0] = u[x1, y0, t1]
            ut1[0, 1] = u[x0, y1, t1]
            ut1[1, 1] = u[x1, y1, t1]

        f0 = ut0[0, 0] + (ut0[1, 0] - ut0[0, 0]) * (x - x0)
        f1 = ut0[0, 1] + (ut0[1, 1] - ut0[0, 1]) * (x - x0)
        finterp0 = f0 + (f1 - f0) * (y - y0)
        f2 = ut1[0, 0] + (ut1[1, 0] - ut1[0, 0]) * (x - x0)
        f3 = ut1[0, 1] + (ut1[1, 1] - ut1[0, 1]) * (x - x0)
        finterp1 = f2 + (f3 - f2) * (y - y0)

        finterp = finterp0 + (finterp1 - finterp0) * (t - t0)

        return finterp

    def Obtain_Gradient(self, x_eval, y_eval, t_eval):
        gradient_array = np.zeros((2, 2))

        gradient_array[0, 0] = self.Interpolate_Fields(x_eval, y_eval, t_eval, self.dudx)
        gradient_array[0, 1] = self.Interpolate_Fields(x_eval, y_eval, t_eval, self.dudy)
        gradient_array[1, 0] = self.Interpolate_Fields(x_eval, y_eval, t_eval, self.dvdx)
        gradient_array[1, 1] = self.Interpolate_Fields(x_eval, y_eval, t_eval, self.dvdy)

        return gradient_array

    def Obtain_Velocity(self, x_eval, y_eval, t_eval):
        velocity_array = np.zeros((2,))

        velocity_array[0] = self.Interpolate_Fields(x_eval, y_eval, t_eval, self.vx)
        velocity_array[1] = self.Interpolate_Fields(x_eval, y_eval, t_eval, self.vy)

        return velocity_array