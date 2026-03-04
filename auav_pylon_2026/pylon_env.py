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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.node = rclpy.create_node('pylon_gym_node')
        
        self.pub = self.node.create_publisher(Joy, '/sim/auto_joy', 10)
        self.sub = self.node.create_subscription(Odometry, '/sim/odom', self._odom_cb, 10)
        
        self.reset_client = self.node.create_client(Empty, '/reset_simulation')
        self.current_odom = None
        
        # Track if we have achieved flight in the current episode
        self.has_taken_off = False 

    def _odom_cb(self, msg):
        self.current_odom = msg

    def _get_obs(self):
        if self.current_odom is None: return np.zeros(6, dtype=np.float32)
        p = self.current_odom.pose.pose.position
        v = self.current_odom.twist.twist.linear
        return np.array([p.x, p.y, p.z, v.x, v.y, v.z], dtype=np.float32)

    def step(self, action):
        joy_msg = Joy()
        joy_msg.axes = [
            float(action[0]), 
            float(action[1]), 
            float(action[2]), 
            float(action[3]), 
            2000.0            
        ]
        self.pub.publish(joy_msg)
        
        rclpy.spin_once(self.node, timeout_sec=0.1)
        obs = self._get_obs()
        
        # Check if we've reached a safe flying altitude (e.g., 1.0 meters)
        if obs[2] > 1.0:
            self.has_taken_off = True

        # Smarter crash logic
        # It crashes if it falls through the map OR hits the ground after taking off
        done = bool((self.has_taken_off and obs[2] < 0.1) or (obs[2] < -0.5))
        
        # Adjusted reward logic
        if done:
            reward = -10.0 # Heavy penalty for crashing
        elif self.has_taken_off:
            reward = 1.0   # Reward for sustaining flight
        else:
            reward = 0.0   # Neutral while accelerating on the runway
        
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        if self.reset_client.wait_for_service(timeout_sec=1.0):
            self.reset_client.call_async(Empty.Request())
        
        self.current_odom = None
        while self.current_odom is None:
            rclpy.spin_once(self.node)
            
        # Reset the flight state for the new episode
        self.has_taken_off = False 
        
        return self._get_obs(), {}