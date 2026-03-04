import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from cognipilot_interfaces.msg import Actuators
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

class PylonRacingEnv(gym.Env):
    def __init__(self):
        super(PylonRacingEnv, self).__init__()
        # Action Space: [Throttle, Aileron, Elevator, Rudder]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # Obs: [x, y, z, vx, vy, vz]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.node = rclpy.create_node('pylon_gym_node')
        self.pub = self.node.create_publisher(Actuators, '/actuators', 10)
        self.sub = self.node.create_subscription(Odometry, '/vehicle_local_position', self._odom_cb, 10)
        self.reset_client = self.node.create_client(Empty, '/reset_simulation')
        self.current_odom = None

    def _odom_cb(self, msg):
        self.current_odom = msg

    def _get_obs(self):
        if self.current_odom is None: return np.zeros(6, dtype=np.float32)
        p = self.current_odom.pose.pose.position
        v = self.current_odom.twist.twist.linear
        return np.array([p.x, p.y, p.z, v.x, v.y, v.z], dtype=np.float32)

    def step(self, action):
        msg = Actuators()
        msg.normalized = action.tolist()
        self.pub.publish(msg)
        
        # Physics Step: Wait 0.1s for simulator to move
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        obs = self._get_obs()
        reward = 1.0 if obs[2] > 0.5 else -1.0 # Simple reward for staying in air
        done = obs[2] < 0.1 # Done if crashed
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.reset_client.call_async(Empty.Request())
        self.current_odom = None
        while self.current_odom is None:
            rclpy.spin_once(self.node)
        return self._get_obs(), {}