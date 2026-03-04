import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

class PylonRacingEnv(gym.Env):
    def __init__(self):
        super(PylonRacingEnv, self).__init__()
        # Action Space: [Aileron, Elevator, Throttle, Rudder]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # Obs: [x, y, z, vx, vy, vz]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.node = rclpy.create_node('pylon_gym_node')
        
        # UPDATED: Match the topics used in the sample script
        self.pub = self.node.create_publisher(Joy, '/sim/auto_joy', 10)
        self.sub = self.node.create_subscription(Odometry, '/sim/odom', self._odom_cb, 10)
        
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
        # UPDATED: Format the action array exactly how the sample brain does
        joy_msg = Joy()
        joy_msg.axes = [
            float(action[0]), # Aileron
            float(action[1]), # Elevator
            float(action[2]), # Throttle
            float(action[3]), # Rudder
            2000.0            # Force onboard stabilizing (from sample script)
        ]
        self.pub.publish(joy_msg)
        
        # Wait for physics update
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        obs = self._get_obs()
        reward = 1.0 if obs[2] > 0.5 else -1.0 # Simple altitude reward
        done = obs[2] < 0.1 # Crashed
        
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        if self.reset_client.wait_for_service(timeout_sec=1.0):
            self.reset_client.call_async(Empty.Request())
        
        self.current_odom = None
        while self.current_odom is None:
            rclpy.spin_once(self.node)
            
        return self._get_obs(), {}