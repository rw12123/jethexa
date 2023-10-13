import mujoco_py
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
from math import acos
import torch as T
import os


register(
    id="Hexapod-v0",
    entry_point="gym.envs.classic_control:Hexapod",
    max_episode_steps=400,
)


class Hexapod(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        # 外部参数
        self.joints_rads_low = np.array([-0.6, -1.2, -0.6] * 6)
        self.joints_rads_high = np.array([0.6, 0.2, 0.6] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.target_vel = 0.4
        self.max_steps = 400

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(46,), dtype=np.float64)

        self.reset()

    def get_state(self):
        return self.sim.get_state()

    def set_state(self, qpos, qvel=None):
        qvel = np.zeros(self.q_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def scale_action(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

    def scale_joints(self, joints):
        return ((np.array(joints) - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1

    def get_agent_obs(self):
        qpos = self.sim.get_state().qpos.tolist()
        qvel = self.sim.get_state().qvel.tolist()

        return np.concatenate((self.scale_joints(qpos[7:]), qvel[6:], qpos[3:7], qvel[:6]))

    def step(self, ctrl):

        ctrl = np.clip(ctrl, -1, 1)

        ctrl_pen = np.square(ctrl).mean()

        ctrl = self.scale_action(ctrl)

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        sx, sy, sz, qw, qx, qy, qz = self.sim.get_state().qpos.tolist()[:7]
        xd, yd, zd, thd, phid, psid = self.sim.get_state().qvel.tolist()[:6]

        velocity_rew = (1. / (abs(xd - self.target_vel) + 1.) - 1.
                        / (self.target_vel + 1.)) * 10
        yaw_rew = np.square(2 * acos(qw)) * .7
        ctrl_pen_rew = np.square(ctrl_pen) * 0.01
        zd_rew = np.square(zd) * 0.5
        r = velocity_rew - yaw_rew - ctrl_pen_rew - zd_rew

        done = self.step_ctr > self.max_steps

        return self.get_agent_obs(), r, False, done, {}

    def reset(self, seed=None, options=None):
        while True:
            try:
                cur_path = os.path.abspath(os.path.dirname(__file__))
                model = "model/hex_flat.xml"
                file_path = os.path.join(cur_path, model)
                self.model = mujoco_py.load_model_from_path(file_path)
                break
            except Exception:
                print("加载模型出错")

        self.sim = mujoco_py.MjSim(self.model)
        self.model.opt.timestep = 0.02
        self.viewer = None

        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = 18 + 18 + 4 + 6
        self.act_dim = self.sim.data.actuator_length.shape[0]

        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = 0.0
        init_q[1] = 0.0
        # init_q[1] = (2*np.random.random()-1)*0.5
        init_q[2] = 0.05
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1
        self.set_state(init_q, init_qvel)
        self.step_ctr = 0

        obs, _, _, _, _ = self.step(np.zeros(self.act_dim))

        return obs, {}

    def setupcam(self):
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.cam.trackbodyid = -1

    def render(self, camera=None):
        if self.viewer is None:
            self.setupcam()
        self.viewer.render()

    def to_tensor(self, x, add_batchdim=False):
        x = T.FloatTensor(x.astype(np.float32))
        if add_batchdim:
            x = x.unsqueeze(0)
        return x

    def test(self, policy, render=True, N=30):
        rew = 0
        for i in range(N):
            obs, _ = self.reset()
            cr = 0
            for j in range(int(self.max_steps)):
                action = policy(self.to_tensor(obs, True)).detach()
                obs, r, done, od, _ = self.step(action[0].numpy())
                cr += r
                rew += r
                if render:
                    self.render()
            print("Total episode reward: {}".format(cr))
        print("Total average reward = {}".format(rew / N))

    def demo(self):
        for i in range(100000):
            self.sim.forward()
            self.sim.step()
            self.render()


if __name__ == "__main__":
    hex = Hexapod()
    hex.demo()


