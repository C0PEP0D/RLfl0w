import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Scripts.Useful import scale_state
from Scripts.Particles import *
import tensorflow as tf
import mayavi.mlab as mlab
from mayavi.mlab import plot3d, volume_slice

plt.rc('text', usetex=True)


def Plot_Vortices_on_Trajectories(*args):
    max_x = []
    max_y = []
    min_x = []
    min_y = []
    x_y_tuple = ()
    labels = []
    for arg in args:
        max_x.append(max(arg[0]))
        max_y.append(max(arg[1]))
        min_x.append(min(arg[0]))
        min_y.append(min(arg[1]))
        x_y_tuple += (arg[0], arg[1])
        labels += [arg[2]]
    max_x = max(max_x)
    max_y = max(max_y)
    min_x = min(min_x)
    min_y = min(min_y)

    x, y = np.meshgrid(np.linspace(min_x - 2 * np.pi, max_x + 2 * np.pi, int(30 * (max_x - min_x))),
                       np.linspace(min_y - 2 * np.pi, max_y + 2 * np.pi, int(30 * (max_y - min_y))))
    TGV_vort = np.cos(x) * np.cos(y)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(x, y, TGV_vort, cmap='RdBu', shading='gouraud')
    cax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.plot(*x_y_tuple)
    ax.set_xlabel('x - axis')
    ax.set_ylabel('y - axis')
    ax.legend(labels)
    ax.set_title("Trajectories in TGV")
    ax.axis('equal')
    ax.set_xlim(-np.pi / 2, 2.5 * np.pi)
    fig.set_figwidth(2.0)

    fig.show()
    return fig, ax


def Plot_Policy(actor, critic, env, scaler, x_con, y_con):
    x, y = np.meshgrid(np.linspace(-np.pi / 2, 3 * np.pi / 2, int(30 * (2 * np.pi))),
                       np.linspace(-np.pi / 2, np.pi / 2, int(30 * (np.pi))))
    TGV_vort = np.cos(x) * np.cos(y)
    x2, y2 = np.meshgrid(np.linspace(-np.pi / 2, 3 * np.pi / 2, x_con), np.linspace(-np.pi / 2, np.pi / 2, y_con))
    states = np.zeros((x2.shape[0], x2.shape[1], env.observation_space.shape[0]))
    dist = []
    actions = np.zeros((x2.shape[0], x2.shape[1]))
    conc = np.zeros((x2.shape[0], x2.shape[1]))
    value = np.zeros((x2.shape[0], x2.shape[1]))
    for i in range(x2.shape[0]):
        dist.append([])
        for j in range(x2.shape[1]):
            states[i, j] = env.pos_to_state(x2[i, j], y2[i, j])
            dist[i].append(actor(scale_state(states[i, j], scaler)))
            if env.type == 'Surf':
                env.particle.current_x = x2[i, j]
                env.particle.current_y = y2[i, j]
                tau = dist[i][j].mean().numpy().reshape(-1)
                # if tau[0] <= 0.0:
                #     tau = np.array([0.0])
                # if tau[0] >= 10.0:
                #     tau = np.array([10.0])
                actions[i, j] = np.float64(
                    env.particle.surf(tau, np.array(env.target_dir)[:, np.newaxis]))
                conc[i, j] = dist[i][j].stddev().numpy().reshape(-1)
            else:
                actions[i, j] = dist[i][j].loc.numpy().reshape(-1)
                conc[i, j] = dist[i][j].concentration.numpy().reshape(-1)
            value[i, j] = critic(scale_state(states[i, j], scaler))

    fig1, axs1 = plt.subplots(figsize=(0.6 * 4.8, 0.6 * 4.8 / 2))
    im = axs1.pcolormesh(x, y, TGV_vort, cmap=plt.cm.get_cmap('RdBu').reversed(), shading='gouraud')
    axs1.quiver(x2, y2, np.cos(actions), np.sin(actions))
    cax = fig1.add_axes([0.95, 0.14, 0.03, 0.72])
    cax.set_title(r'$\omega(s)$', y=1.0)
    fig1.colorbar(im, cax=cax, orientation='vertical', ticks=[TGV_vort.min(), 0.0, TGV_vort.max()])
    axs1.set_xticks([-np.pi / 2, np.pi / 2, 3 * np.pi / 2], labels=['0', '$\pi$', '$2\pi$'])
    axs1.set_yticks([-np.pi / 2, 0.0, np.pi / 2], labels=['0', '$\pi/2$', '$\pi$'])
    # fig1.set_figheight(3.2)
    # axs1.set_title("Policy Representation")
    axs1.axis('equal')
    axs1.set_ylim(-np.pi / 2, np.pi / 2)
    axs1.set_xlim(-np.pi / 2, 3 * np.pi / 2)

    fig2, axs2 = plt.subplots(figsize=(0.6 * 4.8, 0.6 * 4.8 / 2))
    im = axs2.pcolormesh(x2, y2, value, cmap='plasma', shading='gouraud')
    cax = fig2.add_axes([0.95, 0.14, 0.03, 0.72])
    cax.set_title(r'$V(s)$', y=1.0)
    fig2.colorbar(im, cax=cax, orientation='vertical', ticks=[9, 10, 11, 12])
    axs2.set_xticks([-np.pi / 2, np.pi / 2, 3 * np.pi / 2], labels=['0', '$\pi$', '$2\pi$'])
    axs2.set_yticks([-np.pi / 2, 0.0, np.pi / 2], labels=['0', '$\pi/2$', '$\pi$'])
    # fig2.set_figheight(3.2)
    # axs2.set_title("Value Function")
    axs2.axis('equal')
    axs2.set_ylim(-np.pi / 2, np.pi / 2)
    axs2.set_xlim(-np.pi / 2, 3 * np.pi / 2)

    fig3, axs3 = plt.subplots()
    im = axs3.pcolormesh(x2, y2, conc, cmap='viridis')
    cax = fig3.add_axes([0.95, 0.14, 0.02, 0.72])
    if env.type == 'Surf':
        cax.set_title("STD", y=1.0)
        fig3.colorbar(im, cax=cax, orientation='vertical')
        axs3.set_ylim(-np.pi / 2, np.pi / 2)
        fig3.set_figheight(3.2)
        axs3.set_title("Policy Standard Deviation")
        axs3.axis('equal')
    else:
        cax.set_title("conc", y=1.0)
        fig3.colorbar(im, cax=cax, orientation='vertical')
        axs3.set_ylim(-np.pi / 2, np.pi / 2)
        fig3.set_figheight(3.2)
        axs3.set_title("Policy Concentration")
        axs3.axis('equal')

    mean = conc.mean()
    if mean < 0.1:
        done = True
    else:
        done = False
    return [fig1, fig2, fig3], [axs1, axs2, axs3], done


def Plot_Learning_Curve(learning_episodes, average_return):
    fig, ax = plt.subplots()
    ax.plot(learning_episodes, average_return)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Return')
    ax.set_title("Learning curve")
    return fig, ax


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def Plot_Learned_Traj_TGV(actor, scaler, env, N_steps, fig, ax):
    x, y = np.meshgrid(np.linspace(-np.pi / 2, 3 * np.pi / 2, int(30 * (2 * np.pi))),
                       np.linspace(-np.pi / 2, 3 * np.pi / 2 + 2 * np.pi, int(60 * (2 * np.pi))))
    TGV_vort = np.cos(x) * np.cos(y)
    TGV_vort = (TGV_vort - TGV_vort.min()) * 2 / (TGV_vort.max() - TGV_vort.min()) - 1.0
    TGV_vx = -0.5 * np.cos(x) * np.sin(y)
    TGV_vy = 0.5 * np.sin(x) * np.cos(y)
    v_norm = np.sqrt(TGV_vx ** 2 + TGV_vy ** 2)

    im = ax.pcolormesh(x, y, TGV_vort, cmap=plt.cm.get_cmap('RdBu').reversed(), shading='gouraud')

    y_pos = 0.0
    x_part = np.linspace(-np.pi / 2, 3 * np.pi / 2, 14)[1:-1]
    time = np.linspace(0.0, 1.0, N_steps)
    norm = plt.Normalize(time.min(), time.max())
    for x_pos in x_part:
        # PPO part:
        state = env.reset()
        env.particle.current_x = x_pos
        env.particle.current_y = y_pos
        env.particle.history_x = [x_pos]
        env.particle.history_y = [y_pos]
        state = env.pos_to_state(x_pos, y_pos)

        done = False
        while not done:
            act = actor(scale_state(state, scaler))
            action = np.squeeze(act, axis=0)
            state, rew, done, _ = env.step(action)

        traj_x, traj_y = env.particle.get_absolute_positions()
        im2 = ax.scatter(traj_x, traj_y, s=0.2, marker=None, linestyle='-', linewidths=0.2, c=norm(time),
                         cmap='viridis')

    box = ax.get_position()

    cax = fig.add_axes([0.8, box.y0, 0.03, box.height])
    cbar = fig.colorbar(im2, cax=cax, orientation='vertical', ticks=[0.0, np.max(norm(time))], label='Time')
    cbar.ax.set_yticklabels(['$t_{0}$', '$2t_{f}/3$'])

    ax.axhline(y=y_pos, color='black', linestyle='--')
    ax.set_xticks([-3 * np.pi / 2, -np.pi / 2, np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2],
                  labels=['$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$'])
    ax.set_yticks([-np.pi / 2, np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2, 7 * np.pi / 2, 9 * np.pi / 2,
                   11 * np.pi / 2],
                  labels=['0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$', '$5\pi$', '$6\pi$'])
    #     ax.axis('equal')
    ax.set_aspect('equal')
    ax.set_ylim(-np.pi / 2, 7 * np.pi / 2)
    ax.set_xlim(-np.pi / 2 - 0.2, 3 * np.pi / 2 + 0.2)

    plt.show()


def Plot_Naive_Traj_TGV(VelF, N_steps, fig, ax):
    x, y = np.meshgrid(np.linspace(-np.pi / 2, 3 * np.pi / 2, int(30 * (2 * np.pi))),
                       np.linspace(-np.pi / 2, 3 * np.pi / 2 + 2 * np.pi, int(60 * (2 * np.pi))))
    TGV_vort = np.cos(x) * np.cos(y)
    TGV_vort = (TGV_vort - TGV_vort.min()) * 2 / (TGV_vort.max() - TGV_vort.min()) - 1.0
    TGV_vx = -0.5 * np.cos(x) * np.sin(y)
    TGV_vy = 0.5 * np.sin(x) * np.cos(y)
    v_norm = np.sqrt(TGV_vx ** 2 + TGV_vy ** 2)

    im = ax.pcolormesh(x, y, TGV_vort, cmap=plt.cm.get_cmap('RdBu').reversed(), shading='gouraud')

    y_pos = 0.0
    x_part = np.linspace(-np.pi / 2, 3 * np.pi / 2, 14)[1:-1]
    time = np.linspace(0.0, 1.0, N_steps)
    norm = plt.Normalize(time.min(), time.max())
    for x_pos in x_part:
        ActivePart = ActiveParticle(x_pos, y_pos, VelF, np.pi / 2, 0.25)
        fx, fy = ActivePart.Simulate_Trajectory(N_steps, 0.01)
        traj_x, traj_y = ActivePart.get_absolute_positions()
        im2 = ax.scatter(traj_x, traj_y, s=0.2, marker=None, linestyle='-', linewidths=0.2, c=norm(time),
                         cmap='viridis')

    ax.axhline(y=y_pos, color='black', linestyle='--')
    ax.set_xticks([-3 * np.pi / 2, -np.pi / 2, np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2],
                  labels=['$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$'])
    ax.set_yticks([-np.pi / 2, np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2, 7 * np.pi / 2, 9 * np.pi / 2, 11 * np.pi / 2],
                  labels=['0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$', '$5\pi$', '$6\pi$'])

    ax.set_aspect('equal')
    ax.set_ylim(-np.pi / 2, 7 * np.pi / 2)
    ax.set_xlim(-np.pi / 2 - 0.2, 3 * np.pi / 2 + 0.2)


    plt.show()


def Plot_Surf_Traj_TGV(env, N_steps, fig, ax):
    x, y = np.meshgrid(np.linspace(-np.pi / 2, 3 * np.pi / 2, int(30 * (2 * np.pi))),
                       np.linspace(-np.pi / 2, 3 * np.pi / 2 + 2 * np.pi, int(60 * (2 * np.pi))))
    TGV_vort = np.cos(x) * np.cos(y)
    TGV_vort = (TGV_vort - TGV_vort.min()) * 2 / (TGV_vort.max() - TGV_vort.min()) - 1.0
    TGV_vx = -0.5 * np.cos(x) * np.sin(y)
    TGV_vy = 0.5 * np.sin(x) * np.cos(y)
    v_norm = np.sqrt(TGV_vx ** 2 + TGV_vy ** 2)

    im = ax.pcolormesh(x, y, TGV_vort, cmap=plt.cm.get_cmap('RdBu').reversed(), shading='gouraud')

    y_pos = 0.0
    x_part = np.linspace(-np.pi / 2, 3 * np.pi / 2, 14)[1:-1]
    time = np.linspace(0.0, 1.0, N_steps)
    norm = plt.Normalize(time.min(), time.max())
    tau = 2.0
    for x_pos in x_part:
        # PPO part:
        state = env.reset()
        env.particle.current_x = x_pos
        env.particle.current_y = y_pos
        env.particle.history_x = [x_pos]
        env.particle.history_y = [y_pos]
        state = env.pos_to_state(x_pos, y_pos)

        done = False
        while not done:
            action = env.particle.surf(tau, env.target_dir)
            state, rew, done, _ = env.step([action])

        traj_x, traj_y = env.particle.get_absolute_positions()
        im2 = ax.scatter(traj_x, traj_y, s=0.2, marker=None, linestyle='-', linewidths=0.2, c=norm(time),
                         cmap='viridis')

    box = ax.get_position()

    #     cax = fig.add_axes([0.95, box.y0, 0.03, box.height])
    #     cbar = fig.colorbar(im2, cax=cax, orientation='vertical', ticks=[0.0, np.max(norm(time))], label='Time')
    #     cbar.ax.set_yticklabels(['$t_{0}$', '$2t_{f}/3$'])

    ax.axhline(y=y_pos, color='black', linestyle='--')
    ax.set_xticks([-3 * np.pi / 2, -np.pi / 2, np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2],
                  labels=['$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$'])
    ax.set_yticks([-np.pi / 2, np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2, 7 * np.pi / 2, 9 * np.pi / 2,
                   11 * np.pi / 2],
                  labels=['0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$', '$5\pi$', '$6\pi$'])
    #     ax.axis('equal')
    ax.set_aspect('equal')
    ax.set_ylim(-np.pi / 2, 7 * np.pi / 2)
    ax.set_xlim(-np.pi / 2 - 0.2, 3 * np.pi / 2 + 0.2)


    plt.show()


def Plot_Passive_Traj_TGV(VelF, N_steps, fig, ax):
    y_pos = 0.0
    x_part = np.linspace(-np.pi / 2, 3 * np.pi / 2, 14)[1:-1]
    time = np.linspace(0.0, 1.0, N_steps)
    norm = plt.Normalize(time.min(), time.max())
    for x_pos in x_part:
        ActivePart = PassiveParticle(x_pos, y_pos, VelF)
        fx, fy = ActivePart.Simulate_Trajectory(N_steps, 0.01)
        traj_x, traj_y = ActivePart.get_absolute_positions()
        im2 = ax.scatter(traj_x, traj_y, s=0.2, marker=None, linestyle='-', linewidths=0.2, c=norm(time),
                         cmap='viridis')

    ax.axhline(y=y_pos, xmin=0.28, xmax=0.75, color='black', linestyle='--')
    ax.set_xticks([-3 * np.pi / 2, -np.pi / 2, np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2],
                  labels=['$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$'])
    ax.set_yticks([-np.pi / 2, np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2, 7 * np.pi / 2, 9 * np.pi / 2, 11 * np.pi / 2],
                  labels=['0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$', '$5\pi$', '$6\pi$'])

    ax.set_aspect('equal')
    ax.set_ylim(-np.pi / 2, 11 * np.pi / 2)
    ax.set_xlim(-np.pi / 2 - 3.7, 3 * np.pi / 2 + 3.7)


    plt.show()


def Plot_Vorticity_TGV(VelF, fig, ax):
    x, y = np.meshgrid(np.linspace(-np.pi / 2, 3 * np.pi / 2, int(30 * (2 * np.pi))),
                       np.linspace(-np.pi / 2, 3 * np.pi / 2, int(30 * (2 * np.pi))))
    TGV_vort = np.cos(x) * np.cos(y)
    TGV_vort = (TGV_vort - TGV_vort.min()) * 2 / (TGV_vort.max() - TGV_vort.min()) - 1.0
    TGV_vx = -0.5 * np.cos(x) * np.sin(y)
    TGV_vy = 0.5 * np.sin(x) * np.cos(y)
    v_norm = np.sqrt(TGV_vx ** 2 + TGV_vy ** 2)

    im = ax.pcolormesh(x, y, TGV_vort, cmap=plt.cm.get_cmap('RdBu').reversed(), shading='gouraud')
    points = np.array([[0., -1.57],
                       [0., -1.08747438],
                       [0., -0.60415243],
                       [0., 0.36249146],
                       [0., 0.84581341],
                       [0., 1.32913535],
                       [0., 1.8124573],
                       [0., 2.29577925],
                       [0., 2.77910119],
                       [0., 3.74574509],
                       [0., 4.22906703],
                       [0., 4.71238898],
                       [3.14159265, -1.57],
                       [3.14159265, -1.08747438],
                       [3.14159265, -0.60415243],
                       [3.14159265, 0.36249146],
                       [3.14159265, 0.84581341],
                       [3.14159265, 1.32913535],
                       [3.14159265, 1.8124573],
                       [3.14159265, 2.29577925],
                       [3.14159265, 2.77910119],
                       [3.14159265, 3.74574509],
                       [3.14159265, 4.22906703],
                       [3.14159265, 4.71238898]])

    ax.streamplot(x, y, TGV_vx, TGV_vy, density=[1.0, 1.0], color='black',
                  linewidth=0.5 * 2 * v_norm / v_norm.max(), arrowsize=0.5 * 1, broken_streamlines=False,
                  start_points=points)
    #     ax.axhline(y=0.0, color='black', linestyle='--')

    box = ax.get_position()

    cax = fig.add_axes([0.82, box.y0, 0.03, box.height])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[TGV_vort.min(), 0.0, TGV_vort.max()],
                        label=r'$\omega$')
    cbar.ax.set_yticklabels(['$\omega_{min}$', '0', '$\omega_{max}$'])

    ax.set_xticks([-np.pi / 2, np.pi / 2, 3 * np.pi / 2], labels=['0', '$\pi$', '$2\pi$'])
    ax.set_yticks([-np.pi / 2, np.pi / 2, 3 * np.pi / 2], labels=['0', '$\pi$', '$2\pi$'])
    #     ax.set_title("TGV")
    ax.set_aspect('equal')
    ax.set_ylim(-np.pi / 2, 3 * np.pi / 2)
    ax.set_xlim(-np.pi / 2, 3 * np.pi / 2)


    plt.show()


def Plot_Policy_TGV(actor, env, scaler, x_con, y_con, fig, ax):
    x, y = np.meshgrid(np.linspace(-np.pi / 2, 3 * np.pi / 2, int(30 * (2 * np.pi))),
                       np.linspace(-np.pi / 2, 3 * np.pi / 2, int(30 * (2 * np.pi))))
    TGV_vort = np.cos(x) * np.cos(y)
    TGV_vort = (TGV_vort - TGV_vort.min()) * 2 / (TGV_vort.max() - TGV_vort.min()) - 1.0
    TGV_vx = -0.5 * np.cos(x) * np.sin(y)
    TGV_vy = 0.5 * np.sin(x) * np.cos(y)
    v_norm = np.sqrt(TGV_vx ** 2 + TGV_vy ** 2)

    TGV_vy_norm = (TGV_vy - TGV_vy.min()) * 2 / (TGV_vy.max() - TGV_vy.min()) - 1.0

    x2, y2 = np.meshgrid(np.linspace(-np.pi / 2, 3 * np.pi / 2, x_con), np.linspace(-np.pi / 2, 3 * np.pi / 2, y_con))
    states = np.zeros((x2.shape[0], x2.shape[1], env.observation_space.shape[0]))
    dist = []
    actions = np.zeros((x2.shape[0], x2.shape[1]))
    actions_surf = np.zeros((x2.shape[0], x2.shape[1]))
    for i in range(x2.shape[0]):
        dist.append([])
        for j in range(x2.shape[1]):
            env.particle.current_x = x2[i, j]
            env.particle.current_y = y2[i, j]
            states[i, j] = env.pos_to_state(x2[i, j], y2[i, j])
            dist[i].append(actor(scale_state(states[i, j], scaler)))
            actions[i, j] = dist[i][j].loc.numpy().reshape(-1)
            actions_surf[i, j] = env.particle.surf(2.0, env.target_dir)

    cmap = plt.cm.get_cmap('PuOr').reversed()
    new_cmap = truncate_colormap(cmap, 0.1, 0.9, 1000)

    im = ax.pcolormesh(x, y, TGV_vy_norm, cmap=new_cmap, shading='gouraud',
                       vmin=TGV_vy_norm.min(), vmax=TGV_vy_norm.max())
    S2 = ax.streamplot(x2, y2, np.cos(actions_surf), np.sin(actions_surf), density=[0.5, 0.5], color='grey',
                       broken_streamlines=True, linewidth=0.9, arrowsize=0.9)

    S2.lines.set_alpha(1.0)

    for x in ax.get_children():
        if type(x) == matplotlib.patches.FancyArrowPatch:
            x.set_alpha(1.0)

    S1 = ax.streamplot(x2, y2, np.cos(actions), np.sin(actions), density=[0.5, 0.5], color='black',
                       broken_streamlines=True, linewidth=0.9, arrowsize=0.9)

    Q1 = ax.quiver(x2, y2, np.cos(actions), np.sin(actions), alpha=0.0, width=2 * 0.007)
    Q2 = ax.quiver(x2, y2, np.cos(actions_surf), np.sin(actions_surf),
                   color='grey', alpha=0.0, width=2 * 0.007)

    box = ax.get_position()

    cax = fig.add_axes([0.95, box.y0, 0.03, box.height])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[TGV_vy_norm.min(), 0.0, TGV_vy_norm.max()],
                        label='Vertical Velocity')
    cbar.ax.set_yticklabels(['$Uz_{min}$', '0', '$Uz_{max}$'])

    qk1 = ax.quiverkey(Q1, 0.85, 0.6, 5.0, 'PPO', labelpos='E',
                       coordinates='figure', alpha=1.0)
    qk2 = ax.quiverkey(Q2, 0.85, 0.4, 5.0, 'Surf', color='grey', labelpos='E',
                       coordinates='figure', alpha=1.0)

    ax.set_xticks([-np.pi / 2, np.pi / 2, 3 * np.pi / 2], labels=['0', '$\pi$', '$2\pi$'])
    ax.set_yticks([-np.pi / 2, np.pi / 2, 3 * np.pi / 2], labels=['0', '$\pi$', '$2\pi$'])
    ax.set_box_aspect(1.0)
    #     ax.set_title("TGV")
    #     ax.axis('equal')
    ax.set_aspect('equal')
    ax.set_ylim(-np.pi / 2, 3 * np.pi / 2)
    ax.set_xlim(-np.pi / 2, 3 * np.pi / 2)

    plt.show()


def Plot_cube_ABC():
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.))
    plane = np.load('./Scripts/plane.npy')
    plane_x = np.load('./Scripts/plane_x.npy')
    plane_y = np.load('./Scripts/plane_y.npy')
    vmin_c = np.load('./Scripts/vmin_c.npy')
    vmax_c = np.load('./Scripts/vmax_c.npy')

    plane_x = np.stack([plane_x.T] * plane_x.shape[0], axis=0)
    plane_y = np.stack([plane_y.T] * plane_y.shape[0], axis=1)

    val = plane.shape[0] * 1j
    x, y, z = np.mgrid[0.0:2 * np.pi:val, 0.0:2 * np.pi:val, 0.0:2 * np.pi:val]

    obj = volume_slice(x[:, :, 0], y[:, :, 0], z[:, :, 0] + 2 * np.pi, plane.T, plane_orientation='z_axes',
                       colormap='afmhot',
                       vmin=vmin_c, vmax=vmax_c)

    obj = volume_slice(x[:, :, :], y[:, :, :], z[:, :, :], plane_x[:, :, :], plane_orientation='x_axes',
                       colormap='afmhot',
                       vmin=vmin_c, vmax=vmax_c)
    obj = volume_slice(x[:, :, :], y[:, :, :], z[:, :, :], plane_y[:, :, :], plane_orientation='y_axes',
                       colormap='afmhot',
                       vmin=vmin_c, vmax=vmax_c)

    l = plot3d(np.array([2 * np.pi, 2 * np.pi]),
               np.array([0.0, 0.0]),
               np.array([0.0, 2 * np.pi]),
               tube_radius=0.01)

    l = plot3d(np.array([0.0, 2 * np.pi]),
               np.array([0.0, 0.0]),
               np.array([2 * np.pi, 2 * np.pi]),
               tube_radius=0.01)

    l = plot3d(np.array([2 * np.pi, 2 * np.pi]),
               np.array([0.0, 2 * np.pi]),
               np.array([2 * np.pi, 2 * np.pi]),
               tube_radius=0.01)

    mlab.axes(x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True, color=(0, 0, 0),
              ranges=[0, 2 * np.pi, 0, 2 * np.pi, 0.0, 2 * np.pi], nb_labels=3, xlabel='', ylabel='', zlabel='',
              extent=[0, 2 * np.pi, 0, 2 * np.pi, 0.0, 2 * np.pi])

    mlab.view(azimuth=315., elevation=60.)

    mlab.show()


def Plot_Learned_Traj_ABC(actor, scaler, env, N_steps, shift):
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.))
    plane = np.load('./Scripts/plane.npy')
    vmin_c = np.load('./Scripts/vmin_c.npy')
    vmax_c = np.load('./Scripts/vmax_c.npy')

    val = plane.shape[0] * 1j
    x, y, z = np.mgrid[0.0:2 * np.pi:val, 0.0:2 * np.pi:val, 0.0:8 * np.pi:val]
    z_pos = 0.0
    x_part = np.linspace(shift, 2 * np.pi + shift, 12)[1:-1]
    y_part = np.linspace(0.0, 2 * np.pi, 12)[1:-1]
    time = np.linspace(0.0, 1.0, N_steps)

    # obj = volume_slice(x[:, :, 0], y[:, :, 0], z[:, :, 0] , z[:, :, 0]+0.5, plane_orientation='z_axes', vmin=0.0, vmax=1.0 , colormap='binary')
    # obj = volume_slice(x[:,:,0], y[:,:,0], z[:,:,0]-np.pi/2, plane, plane_orientation='z_axes', colormap='afmhot', vmin = vmin_c, vmax=vmax_c)
    obj = volume_slice(x[:, :, 0], y[:, :, 0], z[:, :, 0], plane.T, plane_orientation='z_axes', colormap='afmhot',
                       vmin=vmin_c, vmax=vmax_c)
    obj = volume_slice(x[:, :, 0], y[:, :, 0], z[:, :, int(plane.shape[0] / 4)], plane.T, plane_orientation='z_axes',
                       colormap='afmhot', vmin=vmin_c, vmax=vmax_c)
    obj = volume_slice(x[:, :, 0], y[:, :, 0], z[:, :, int(plane.shape[0] / 2)], plane.T, plane_orientation='z_axes',
                       colormap='afmhot', vmin=vmin_c, vmax=vmax_c)
    obj = volume_slice(x[:, :, 0], y[:, :, 0], z[:, :, int(3 * plane.shape[0] / 4)], plane.T,
                       plane_orientation='z_axes', colormap='afmhot', vmin=vmin_c, vmax=vmax_c)

    for x_pos in x_part:
        for y_pos in y_part:
            # PPO part:
            state = env.reset()
            env.particle.current_x = x_pos
            env.particle.current_y = y_pos
            env.particle.current_z = z_pos
            env.particle.history_x = [x_pos]
            env.particle.history_y = [y_pos]
            env.particle.history_z = [z_pos]
            state = env.pos_to_state(env.particle.current_x, env.particle.current_y, env.particle.current_z)

            done = False
            while not done:
                act = actor(scale_state(state, scaler))
                action = np.squeeze(act, axis=0)
                state, rew, done, _ = env.step(action)

            traj_x, traj_y, traj_z = env.particle.get_absolute_positions()
            # l = plot3d(np.array(env.particle.history_x[:-1]) - shift, np.array(env.particle.history_y[:-1]),
            #                               np.array(traj_z), time, tube_radius=0.025, colormap='viridis')

            x_conf = np.array(env.particle.history_x[:-1]) - shift
            y_conf = np.array(env.particle.history_y[:-1])
            cutting_inds = [0]
            for ind, val_x in enumerate(x_conf[:-1]):
                if np.abs(x_conf[ind + 1] - val_x) >= 2.0 or np.abs(y_conf[ind + 1] - y_conf[ind]) >= 2.0:
                    cutting_inds.append(ind)

            for i in range(len(cutting_inds)):
                if i == len(cutting_inds) - 1:
                    l = plot3d(x_conf[cutting_inds[i] + 1:], y_conf[cutting_inds[i] + 1:],
                               np.array(traj_z)[cutting_inds[i] + 1:], time[cutting_inds[i] + 1:], tube_radius=0.025,
                               colormap='viridis', vmin=time[0], vmax=time[-1])
                elif i == 0:
                    l = plot3d(x_conf[cutting_inds[i]:cutting_inds[i + 1] + 1],
                               y_conf[cutting_inds[i]:cutting_inds[i + 1] + 1],
                               np.array(traj_z)[cutting_inds[i]:cutting_inds[i + 1] + 1],
                               time[cutting_inds[i]:cutting_inds[i + 1] + 1], tube_radius=0.025, colormap='viridis',
                               vmin=time[0], vmax=time[-1])
                else:
                    l = plot3d(x_conf[cutting_inds[i] + 1:cutting_inds[i + 1] + 1],
                               y_conf[cutting_inds[i] + 1:cutting_inds[i + 1] + 1],
                               np.array(traj_z)[cutting_inds[i] + 1:cutting_inds[i + 1] + 1],
                               time[cutting_inds[i] + 1:cutting_inds[i + 1] + 1], tube_radius=0.025, colormap='viridis',
                               vmin=time[0], vmax=time[-1])

    mlab.axes(x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True, color=(0, 0, 0),
              ranges=[0, 2 * np.pi, 0, 2 * np.pi, 0.0, 8 * np.pi], nb_labels=3, xlabel='', ylabel='', zlabel='',
              extent=[0, 2 * np.pi, 0, 2 * np.pi, 0.0, 8 * np.pi])

    mlab.view(azimuth=315., elevation=60.)

    mlab.show()


def Plot_Naive_Traj_ABC(VelF, N_steps, shift):
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.))
    plane = np.load('./Scripts/plane.npy')
    vmin_c = np.load('./Scripts/vmin_c.npy')
    vmax_c = np.load('./Scripts/vmax_c.npy')

    val = plane.shape[0] * 1j
    x, y, z = np.mgrid[0.0:2 * np.pi:val, 0.0:2 * np.pi:val, 0.0:8 * np.pi:val]
    z_pos = 0.0
    x_part = np.linspace(shift, 2 * np.pi + shift, 12)[1:-1]
    y_part = np.linspace(0.0, 2 * np.pi, 12)[1:-1]
    time = np.linspace(0.0, 1.0, N_steps)

    obj = volume_slice(x[:, :, 0], y[:, :, 0], z[:, :, 0], plane.T, plane_orientation='z_axes', colormap='afmhot',
                       vmin=vmin_c, vmax=vmax_c)
    obj = volume_slice(x[:, :, 0], y[:, :, 0], z[:, :, int(plane.shape[0] / 4)], plane.T,
                       plane_orientation='z_axes', colormap='afmhot', vmin=vmin_c, vmax=vmax_c)
    obj = volume_slice(x[:, :, 0], y[:, :, 0], z[:, :, int(plane.shape[0] / 2)], plane.T,
                       plane_orientation='z_axes', colormap='afmhot', vmin=vmin_c, vmax=vmax_c)
    obj = volume_slice(x[:, :, 0], y[:, :, 0], z[:, :, int(3 * plane.shape[0] / 4)], plane.T,
                       plane_orientation='z_axes', colormap='afmhot', vmin=vmin_c, vmax=vmax_c)

    for x_pos in x_part:
        for y_pos in y_part:

            ActivePart = ActiveParticle3D(x_pos, y_pos, z_pos, VelF, 0.0, 0.0, 1.5)
            fx, fy, fz = ActivePart.Simulate_Trajectory(N_steps, 0.01)
            traj_x, traj_y, traj_z = ActivePart.get_absolute_positions()

            x_conf = np.array(ActivePart.history_x[:-1]) - shift
            y_conf = np.array(ActivePart.history_y[:-1])
            cutting_inds = [0]
            for ind, val_x in enumerate(x_conf[:-1]):
                if np.abs(x_conf[ind + 1] - val_x) >= 2.0 or np.abs(y_conf[ind + 1] - y_conf[ind]) >= 2.0:
                    cutting_inds.append(ind)

            for i in range(len(cutting_inds)):
                if i == len(cutting_inds) - 1:
                    l = plot3d(x_conf[cutting_inds[i] + 1:], y_conf[cutting_inds[i] + 1:],
                               np.array(traj_z)[cutting_inds[i] + 1:], time[cutting_inds[i] + 1:],
                               tube_radius=0.025, colormap='viridis', vmin=time[0], vmax=time[-1])
                elif i == 0:
                    l = plot3d(x_conf[cutting_inds[i]:cutting_inds[i + 1] + 1],
                               y_conf[cutting_inds[i]:cutting_inds[i + 1] + 1],
                               np.array(traj_z)[cutting_inds[i]:cutting_inds[i + 1] + 1],
                               time[cutting_inds[i]:cutting_inds[i + 1] + 1], tube_radius=0.025, colormap='viridis',
                               vmin=time[0], vmax=time[-1])
                else:
                    l = plot3d(x_conf[cutting_inds[i] + 1:cutting_inds[i + 1] + 1],
                               y_conf[cutting_inds[i] + 1:cutting_inds[i + 1] + 1],
                               np.array(traj_z)[cutting_inds[i] + 1:cutting_inds[i + 1] + 1],
                               time[cutting_inds[i] + 1:cutting_inds[i + 1] + 1], tube_radius=0.025,
                               colormap='viridis', vmin=time[0], vmax=time[-1])

    mlab.axes(x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True, color=(0, 0, 0),
              ranges=[0, 2 * np.pi, 0, 2 * np.pi, 0.0, 8 * np.pi], nb_labels=3, xlabel='', ylabel='', zlabel='',
              extent=[0, 2 * np.pi, 0, 2 * np.pi, 0.0, 8 * np.pi])

    mlab.view(azimuth=315., elevation=60.)

    mlab.show()


def Plot_Policy_ABC(actor, env, scaler, x_con, y_con, fig, ax):
    shift = 2.0
    displacement = np.load('./Scripts/displacement.npy')
    vmin = np.load('./Scripts/vmin.npy')
    vmax = np.load('./Scripts/vmax.npy')
    points = np.linspace(0, 2 * np.pi, 500)
    x2, y2 = np.meshgrid(np.linspace(shift, 2 * np.pi + shift, x_con), np.linspace(0.0, 2 * np.pi, y_con))
    z2 = 0.0
    states = np.zeros((x2.shape[0], x2.shape[1], env.observation_space.shape[0]))
    dist = []
    actions = np.zeros((x2.shape[0], x2.shape[1], 3))
    actions_surf = np.zeros((x2.shape[0], x2.shape[1], 3))

    for i in range(x2.shape[0]):
        dist.append([])
        for j in range(x2.shape[1]):
            env.particle.current_x = x2[i, j]
            env.particle.current_y = y2[i, j]
            env.particle.current_z = z2
            states[i, j] = env.pos_to_state(x2[i, j], y2[i, j], z2)
            dist[i].append(actor(scale_state(states[i, j], scaler)))

            actions[i, j] = dist[i][j].mean_direction.numpy().reshape(-1)

            actions_surf[i, j] = env.particle.surf(1.0, env.target_dir)

    cmap = cmap = plt.cm.get_cmap('afmhot').reversed()
    cmap_new = truncate_colormap(cmap, 0.1, 0.9, 1000)
    im = ax.contourf(points, points, displacement, cmap=cmap_new,
                     vmin=displacement.min(), vmax=displacement.max(), levels=1000)

    S2 = ax.streamplot(x2 - shift, y2, actions_surf[:, :, 0], actions_surf[:, :, 1], density=[0.5, 0.5], color='grey',
                       broken_streamlines=True, linewidth=0.9, arrowsize=0.9)

    S2.lines.set_alpha(1.0)
    for x in ax.get_children():
        if type(x) == matplotlib.patches.FancyArrowPatch:
            x.set_alpha(1.0)

    S1 = ax.streamplot(x2 - shift, y2, actions[:, :, 0], actions[:, :, 1], density=[0.5, 0.5], color='black',
                       broken_streamlines=True, linewidth=0.9, arrowsize=0.9)

    Q1 = ax.quiver(x2, y2, actions[:, :, 0], actions[:, :, 1], alpha=0.0, width=2 * 0.007)
    Q2 = ax.quiver(x2, y2, actions_surf[:, :, 0], actions_surf[:, :, 1],
                   color='grey', alpha=0.0, width=2 * 0.007)

    box = ax.get_position()

    cax = fig.add_axes([0.95, box.y0, 0.03, box.height])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[displacement.min(), 0.0, displacement.max()],
                        label='Distance Traveled')
    cbar.ax.set_yticklabels(['$|\Delta X|_{min}$', '0', '$|\Delta X|_{max}$'])

    qk1 = ax.quiverkey(Q1, 0.85, 0.6, 5.0, 'PPO', labelpos='E',
                       coordinates='figure', alpha=1.0)
    qk2 = ax.quiverkey(Q2, 0.85, 0.4, 5.0, 'Surf', color='grey', labelpos='E',
                       coordinates='figure', alpha=1.0)

    ax.set_ylim(0.0, 2 * np.pi)
    ax.set_xlim(0.0, 2 * np.pi)
    ax.set_xticks([0.0, np.pi, 2 * np.pi], labels=['0', '$\pi$', '$2\pi$'])
    ax.set_yticks([0.0, np.pi, 2 * np.pi], labels=['0', '$\pi$', '$2\pi$'])
    #     ax.set_yticks([])
    ax.set_box_aspect(1.0)
    #     ax.set_title("ABC")
    #     ax.axis('equal')
    ax.set_aspect('equal')

    plt.show()


def Plot_Learned_Traj_Turb(actor, scaler, env, t, N_steps, fig, ax, lstm=False):
    y_pos = np.pi / 2
    x_part = np.linspace(0, 2 * np.pi, 22)[1:-1]
    time = np.linspace(0.0, 1.0, N_steps)
    norm = plt.Normalize(time.min(), time.max())
    for x_pos in x_part:
        # PPO part:
        state = env.reset()
        env.particle.current_x = x_pos
        env.particle.current_y = y_pos
        env.particle.current_t = t
        env.particle.history_x = [x_pos]
        env.particle.history_y = [y_pos]
        env.particle.history_t = [float(t)]
        state = env.pos_to_state(x_pos, y_pos, t)

        done = False
        if lstm:
            actor.reset_states()
            while not done:
                act = actor(tf.expand_dims(scale_state(state, scaler), axis=1))
                action = np.squeeze(np.squeeze(act, axis=0), axis=0)
                state, rew, done, _ = env.step(action)
            actor.reset_states()
        else:
            while not done:
                act = actor(scale_state(state, scaler))
                action = np.squeeze(act, axis=0)
                state, rew, done, _ = env.step(action)

        traj_x, traj_y = env.particle.get_absolute_positions()
        im2 = ax.scatter(traj_x, traj_y, s=0.3, marker=None, linestyle='-', linewidths=0.4, c=norm(time),
                         cmap='viridis')
    #         im2 = ax.scatter(env.particle.history_x[:-1], traj_y, s=0.2, marker=None, linestyle='-', linewidths=0.2, c=norm(time), cmap= 'viridis')

    ax.axhline(y=y_pos, xmin=0.0, xmax=1.0, color='black', linestyle='--')
    ax.set_xticks([-2 * np.pi, -np.pi, 0.0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi],
                  labels=['$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'])
    ax.set_yticks(
        [-3 * np.pi, -2 * np.pi, -np.pi, 0.0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi, 7 * np.pi],
        labels=['$-3\pi$', '$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$', '$5\pi$', '$6\pi$',
                '$7\pi$'])
    #     ax.axis('equal')
    ax.set_aspect('equal')
    ax.set_ylim(0.0, 4 * np.pi)
    ax.set_xlim(0.0, 2 * np.pi)

    box = ax.get_position()

    cax = fig.add_axes([0.8, box.y0, 0.03, box.height])
    cbar = fig.colorbar(im2, cax=cax, orientation='vertical', ticks=[0.0, np.max(norm(time))], label='Time')
    cbar.ax.set_yticklabels(['$t_{0}$', '$7t_{f}/12$'])

    plt.show()


def Plot_Naive_Traj_Turb(VelF, t, N_steps, fig, ax):
    y_pos = np.pi / 2
    x_part = np.linspace(0, 2 * np.pi, 22)[1:-1]
    time = np.linspace(0.0, 1.0, N_steps)
    norm = plt.Normalize(time.min(), time.max())
    for x_pos in x_part:
        ActivePart = ActiveParticleTran(x_pos, y_pos, t, VelF, np.pi / 2, 2.0)
        fx, fy = ActivePart.Simulate_Trajectory(N_steps, 0.01)
        traj_x, traj_y = ActivePart.get_absolute_positions()

        im2 = ax.scatter(traj_x, traj_y, s=0.4, marker=None, linestyle='-', linewidths=0.3, c=norm(time),
                         cmap='viridis')
    #         im2 = ax.scatter(ActivePart.history_x[:-1], traj_y, s=0.2, marker=None, linestyle='-', linewidths=0.2, c=norm(time), cmap= 'viridis')

    #     ax.axhline(y=y_pos, color='black', linestyle='--')
    ax.axhline(y=y_pos, xmin=0.0, xmax=1.0, color='black', linestyle='--')
    ax.set_xticks([-2 * np.pi, -np.pi, 0.0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi],
                  labels=['$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'])
    ax.set_yticks(
        [-3 * np.pi, -2 * np.pi, -np.pi, 0.0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi, 7 * np.pi],
        labels=['$-3\pi$', '$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$', '$5\pi$', '$6\pi$',
                '$7\pi$'])
    ax.axis('equal')
    ax.set_aspect('equal')
    ax.set_ylim(0.0, 4 * np.pi)
    ax.set_xlim(0.0, 2 * np.pi)

    #     ax.set_aspect('equal')
    #     ax.set_ylim(0.0, 4*np.pi)
    #     ax.set_xlim(0.0 ,2*np.pi)

    plt.show()


def Plot_Surf_Traj_Turb(env, t, N_steps, fig, ax):
    y_pos = np.pi / 2
    x_part = np.linspace(0, 2 * np.pi, 22)[1:-1]
    time = np.linspace(0.0, 1.0, N_steps)
    norm = plt.Normalize(time.min(), time.max())
    tau = 0.23
    for x_pos in x_part:
        # PPO part:
        state = env.reset()
        env.particle.current_x = x_pos
        env.particle.current_y = y_pos
        env.particle.current_t = t
        env.particle.history_x = [x_pos]
        env.particle.history_y = [y_pos]
        env.particle.history_t = [float(t)]
        state = env.pos_to_state(x_pos, y_pos, t)
        done = False
        while not done:
            action = env.particle.surf(tau, env.target_dir)
            state, rew, done, _ = env.step([action])

        traj_x, traj_y = env.particle.get_absolute_positions()
        im2 = ax.scatter(traj_x, traj_y, s=0.4, marker=None, linestyle='-', linewidths=0.3, c=norm(time),
                         cmap='viridis')

    ax.axhline(y=y_pos, xmin=0.0, xmax=1.0, color='black', linestyle='--')
    ax.set_xticks([-2 * np.pi, -np.pi, 0.0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi],
                  labels=['$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'])
    ax.set_yticks(
        [-3 * np.pi, -2 * np.pi, -np.pi, 0.0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi, 7 * np.pi],
        labels=['$-3\pi$', '$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$', '$5\pi$', '$6\pi$',
                '$7\pi$'])
    #     ax.axis('equal')
    ax.set_aspect('equal')
    ax.set_ylim(0.0, 4 * np.pi)
    ax.set_xlim(0.0, 2 * np.pi)

    plt.show()


def Plot_Passive_Traj_Turb(VelF, t, N_steps, fig, ax):
    y_pos = np.pi / 2
    x_part = np.linspace(0, 2 * np.pi, 22)[1:-1]
    time = np.linspace(0.0, 1.0, N_steps)
    norm = plt.Normalize(time.min(), time.max())
    for x_pos in x_part:
        ActivePart = PassiveParticleTran(x_pos, y_pos, t, VelF)
        fx, fy = ActivePart.Simulate_Trajectory(N_steps, 0.01)
        traj_x, traj_y = ActivePart.get_absolute_positions()

        im2 = ax.scatter(traj_x, traj_y, s=0.2, marker=None, linestyle='-', linewidths=0.2, c=norm(time),
                         cmap='viridis')

    ax.axhline(y=y_pos, xmin=0.35, xmax=0.65, color='black', linestyle='--')
    ax.set_xticks([-2 * np.pi, -np.pi, 0.0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi],
                  labels=['$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'])
    ax.set_yticks(
        [-3 * np.pi, -2 * np.pi, -np.pi, 0.0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi, 7 * np.pi],
        labels=['$-3\pi$', '$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$', '$5\pi$', '$6\pi$',
                '$7\pi$'])
    #     ax.axis('equal')
    ax.axis('equal')
    ax.set_aspect('equal')
    ax.set_ylim(-3 * np.pi, 7 * np.pi)
    ax.set_xlim(-2 * np.pi - 2.0, 4 * np.pi + 2.0)

    plt.show()


def Plot_Vorticity_Turb(VelF, t, fig, ax):
    x, y = np.meshgrid(np.linspace(0.0, 2 * np.pi, int(30 * (2 * np.pi))),
                       np.linspace(0.0, 2 * np.pi, int(30 * (2 * np.pi))))
    vort_z = np.zeros((x.shape[0], x.shape[1]))
    u_x = np.zeros((x.shape[0], x.shape[1]))
    u_y = np.zeros((x.shape[0], x.shape[1]))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vort_z[i, j] = VelF.Interpolate_Fields(x[i, j], y[i, j], t, VelF.vort_z)
            u_x[i, j] = VelF.Interpolate_Fields(x[i, j], y[i, j], t, VelF.vx)
            u_y[i, j] = VelF.Interpolate_Fields(x[i, j], y[i, j], t, VelF.vy)

    vort_z = (vort_z - vort_z.min()) * 2 / (vort_z.max() - vort_z.min()) - 1.0
    v_norm = np.sqrt(u_x ** 2 + u_y ** 2)

    im = ax.pcolormesh(x, y, vort_z, cmap=plt.cm.get_cmap('RdBu').reversed(), shading='gouraud')
    y_part = np.linspace(0.0, 2 * np.pi, 10)
    x_part = np.linspace(0.0, 2 * np.pi, 10)

    ax.streamplot(x, y, u_x, u_y, density=[1.0, 1.0], color='black',
                  linewidth=0.5 * 3 * v_norm / v_norm.max(), arrowsize=0.5, broken_streamlines=True)

    #     ax.axhline(y=np.pi/2, color='black', linestyle='--')

    ax.set_xticks([0.0, np.pi, 2 * np.pi], labels=['0', '$\pi$', '$2\pi$'])
    ax.set_yticks([0.0, np.pi, 2 * np.pi], labels=['0', '$\pi$', '$2\pi$'])
    #     ax.set_yticks([])

    #     ax.set_yticks([])
    ax.set_box_aspect(1.0)
    #     ax.set_title("TURB")
    ax.axis('equal')
    ax.set_aspect('equal')
    ax.set_ylim(0.0, 2 * np.pi)
    ax.set_xlim(0.0, 2 * np.pi)

    box = ax.get_position()

    cax = fig.add_axes([0.82, box.y0, 0.03, box.height])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[vort_z.min(), 0.0, vort_z.max()])
    cbar.ax.set_yticklabels(['$\omega_{min}$', '0', '$\omega_{max}$'])

    plt.show()


def Plot_Policy_Turb(actor, env, scaler, x_con, y_con, time, fig, ax):
    x, y = np.meshgrid(np.linspace(0.0, 2 * np.pi, int(30 * (2 * np.pi))),
                       np.linspace(0.0, 2 * np.pi, int(30 * (2 * np.pi))))
    vort_z = np.zeros((x.shape[0], x.shape[1]))
    u_y = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vort_z[i, j] = env.velf.Interpolate_Fields(x[i, j], y[i, j], time, env.velf.vort_z)
            u_y[i, j] = env.velf.Interpolate_Fields(x[i, j], y[i, j], time, env.velf.vy)

    u_y = (u_y - u_y.min()) * 2 / (u_y.max() - u_y.min()) - 1.0
    x2, y2 = np.meshgrid(np.linspace(0.0, 2 * np.pi, x_con), np.linspace(0.0, 2 * np.pi, y_con))
    states = np.zeros((x2.shape[0], x2.shape[1], env.observation_space.shape[0]))
    actions = np.zeros((x2.shape[0], x2.shape[1]))
    actions_surf = np.zeros((x2.shape[0], x2.shape[1]))
    dist = []
    for i in range(x2.shape[0]):
        dist.append([])
        for j in range(x2.shape[1]):
            states[i, j] = env.pos_to_state(x2[i, j], y2[i, j], time)
            dist[i].append(actor(scale_state(states[i, j], scaler)))
            actions[i, j] = dist[i][j].loc.numpy().reshape(-1)
            env.particle.current_x = x2[i, j]
            env.particle.current_y = y2[i, j]
            env.particle.current_t = time
            actions_surf[i, j] = np.float64(env.particle.surf(0.25, np.array(env.target_dir)[:, np.newaxis]))

    cmap = plt.cm.get_cmap('PuOr').reversed()
    new_cmap = truncate_colormap(cmap, 0.1, 0.9, 1000)
    im = ax.pcolormesh(x, y, u_y, cmap=new_cmap, shading='gouraud', vmin=u_y.min(), vmax=u_y.max())

    S2 = ax.streamplot(x2, y2, np.cos(actions_surf), np.sin(actions_surf), density=[0.7, 0.7], color='grey',
                       broken_streamlines=True, linewidth=0.9, arrowsize=0.9)
    S2.lines.set_alpha(1.0)

    for x in ax.get_children():
        if type(x) == matplotlib.patches.FancyArrowPatch:
            x.set_alpha(1.0)

    S1 = ax.streamplot(x2, y2, np.cos(actions), np.sin(actions), density=[0.7, 0.7], color='black',
                       broken_streamlines=True, linewidth=0.9, arrowsize=0.9)

    Q1 = ax.quiver(x2, y2, np.cos(actions), np.sin(actions), alpha=0.0, width=2 * 0.007)
    Q2 = ax.quiver(x2, y2, np.cos(actions_surf), np.sin(actions_surf),
                   color='grey', alpha=0.0, width=2 * 0.007)

    box = ax.get_position()

    cax = fig.add_axes([0.95, box.y0, 0.03, box.height])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[u_y.min(), 0.0, u_y.max()],
                        label='Vertical Velocity')
    cbar.ax.set_yticklabels(['$Uz_{min}$', '0', '$Uz_{max}$'])

    qk1 = ax.quiverkey(Q1, 0.85, 0.6, 5.0, 'PPO', labelpos='E',
                       coordinates='figure', alpha=1.0)
    qk2 = ax.quiverkey(Q2, 0.85, 0.4, 5.0, 'Surf', color='grey', labelpos='E',
                       coordinates='figure', alpha=1.0)

    ax.set_xticks([0.0, np.pi, 2 * np.pi], labels=['0', '$\pi$', '$2\pi$'])
    ax.set_yticks([0.0, np.pi, 2 * np.pi], labels=['0', '$\pi$', '$2\pi$'])
    ax.set_box_aspect(1.0)
    #     ax.set_title("TURB")
    #     ax.axis('equal')
    ax.set_aspect('equal')
    ax.set_ylim(0.0, 2 * np.pi)
    ax.set_xlim(0.0, 2 * np.pi)

    plt.show()
