import numpy as np

from robopal.demos.manipulation_tasks.robot_manipulate_dense import ManipulateDenseEnv
import robopal.commons.transform as T
from robopal.robots.ur5e import UR5eReach


def one_hot_encode(task_id, num_tasks):
    one_hot_vector = np.zeros(num_tasks)
    one_hot_vector[task_id] = 1
    return one_hot_vector
class ReachHandlingEnv(ManipulateDenseEnv):

    def __init__(self,
                 robot=UR5eReach,
                 render_mode='human',
                 control_freq=20,
                 enable_camera_viewer=False,
                 controller='CARTIK',
                 task_id=0,
                 num_tasks=4,
                 task_sequence=None  # 任务序列
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            controller=controller,
        )
        self.name = 'ReachTask-v2'
        self.task_id = task_id
        self.task_one_hot = one_hot_encode(task_id, num_tasks)  # 初始化任务 One-Hot 编码
        self.num_tasks = num_tasks
        self.task_sequence = task_sequence if task_sequence else [0, 1, 2, 3]  # 任务序列
        self.current_task_idx = 0  # 当前任务索引

        self.obs_dim = (22 + num_tasks,)  # 更新观测空间维度以包含 One-Hot 编码
        self.goal_dim = (3,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 500

        self.pos_max_bound = np.array([0.7, 0.6, 0.3])
        self.pos_min_bound = np.array([-0.4, -0.7, -0.13])
        self.grip_max_bound = 0.95
        self.grip_min_bound = 0.0

    def step(self, action):
        # 调用父类的 step 方法
        obs, reward, terminated, truncated, info = super().step(action)

        # 确保 info 是一个字典，如果它不是，初始化它
        if not isinstance(info, dict):
            info = {}

        # 将当前任务 ID 设置到 info 字典中
        info['task_id'] = self.task_id

        # 计算 gripper 和 green_block 之间的距离
        dis_cubegripper = self.goal_distance(self.get_body_pos('green_block'), self.get_site_pos('0_grip_site'))

        # 根据距离设置 terminated 条件
        if dis_cubegripper <= 0.04:  # 如果 gripper 接近 green_block
            terminated = True
        if self.get_body_pos('green_block')[2] < 0.1:
            terminated = True

        # 调用 _get_info() 来获取额外的信息并更新到 info 字典中
        # info.update(self._get_info())

        return obs, reward, terminated, truncated, info


    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        obs = np.zeros(self.obs_dim)

        obs[0:3] = (  # gripper position in global coordinates
            end_pos := self.get_site_pos('0_grip_site')
        )
        obs[3:6] = (  # block position in global coordinates
            object_pos := self.get_body_pos('green_block')
        )
        obs[6:9] = (  # Relative block position with respect to gripper position in globla coordinates.
            end_pos - object_pos
        )
        obs[9:12] = (  # block rotation
            T.mat_2_euler(self.get_body_rotm('green_block'))
        )
        obs[12:15] = (  # gripper linear velocity
            self.get_site_xvelp('0_grip_site') * self.dt
        )
        obs[15:18] = (  # block linear velocity
            self.get_body_xvelp('green_block') * self.dt
        )
        obs[18:21] = (  # block angular velocity
            self.get_body_xvelr('green_block') * self.dt
        )
        obs[21] = self.mj_data.joint('0_robotiq_2f_85_right_driver_joint').qpos[0]

        obs[22:] = one_hot_encode(self.task_sequence[self.current_task_idx], len(self.task_sequence))

        return obs.copy()

    def _get_info(self) -> dict:
        return {'is_success': self._is_success(self.get_body_pos('green_block'), self.get_body_pos('carton'), th=0.02)}

    def reset_object(self):
        random_x_pos = np.random.uniform(1.6, 1.7)
        self.set_object_pose('green_block:joint', np.array([random_x_pos, 0.3, 0.75, 1.0, 0.0, 0.0, 0.0]))

    def compute_rewards(self, info: dict = None, **kwargs):

        cube2gripper = self.goal_distance(self.get_body_pos('green_block'), self.get_site_pos('0_grip_site'))
        reward = -100* cube2gripper  # give higher reward the closer the gripper to the cube

        return reward


if __name__ == "__main__":
    task_id = 0
    env = ReachHandlingEnv()
    env.reset()

    for t in range(int(1e5)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, info = env.step(action)
        if terminated:
            env.reset()
    env.close()
