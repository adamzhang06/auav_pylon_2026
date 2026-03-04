#!/usr/bin/env python3
import rclpy
import gymnasium as gym
import numpy as np
# This import works because of the __init__.py we discussed
from auav_pylon_2026.pylon_env import PylonRacingEnv 

def main():
    # 1. Initialize ROS 2
    rclpy.init()
    
    # 2. Setup the Environment
    print("Connecting to Pylon Racing Simulation...")
    env = PylonRacingEnv()

    # 3. Perform a Reset
    print("Testing Reset (Drone should teleport to start)...")
    obs, info = env.reset()
    print(f"Initial State: {obs}")

    # 4. Run 50 random steps to see the drone "twitch"
    print("Executing random actions for 5 seconds...")
    for i in range(50):
        # Random inputs for [Throttle, Aileron, Elevator, Rudder]
        action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i} | Alt: {obs[2]:.2f}m | Reward: {reward:.2f}")

        if terminated:
            print("Crashed! Resetting...")
            env.reset()

    print("Test Complete. Check Gazebo/RViz for movement.")
    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()