#!/usr/bin/env python3
import rclpy
import numpy as np
from auav_pylon_2026.pylon_env import PylonRacingEnv 

def main():
    rclpy.init()
    print("Connecting to Pylon Racing Simulation...")
    env = PylonRacingEnv()

    print("Testing Reset...")
    obs, info = env.reset()
    
    target_alt = 7.0
    print(f"Initiating Takeoff Sequence to {target_alt}m...")

    # Run for 200 steps (approx 20 seconds of simulation time)
    for i in range(200):
        # Extract data from observation: [x, y, z, vx, vy, vz]
        current_alt = obs[2]
        current_speed = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)

        # Default controls: stay straight, full power
        aileron = 0.0
        rudder = 0.0
        throttle = 1.0 
        
        # Hardcoded Takeoff Policy (mimicking the sample script)
        if current_speed < 0.5:
            # Keep tail down while accelerating
            elevator = -0.02
        elif current_alt < target_alt:
            # Pitch up to climb
            elevator = 0.15
        else:
            # Level off once we hit 7 meters
            elevator = 0.0
            throttle = 0.6 # Cruise power

        # Construct the action array: [Aileron, Elevator, Throttle, Rudder]
        action = np.array([aileron, elevator, throttle, rudder], dtype=np.float32)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i} | Speed: {current_speed:.2f} m/s | Alt: {current_alt:.2f} m | Elev: {elevator}")

        if terminated:
            print("Crashed! Resetting...")
            obs, info = env.reset()

    print("Takeoff Test Complete. Did it fly?")
    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()