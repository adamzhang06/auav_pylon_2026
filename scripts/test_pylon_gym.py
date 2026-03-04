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

    # Variable to hold our current elevator state so we can ramp it smoothly
    current_elevator = -0.02 

    for i in range(1000):
        current_alt = obs[2]
        current_speed = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)

        aileron = 0.0
        rudder = 0.0
        throttle = 1.0 
        
        # Smoother Takeoff Policy
        if current_speed < 0.5:
            current_elevator = -0.02 # Tail down
        elif current_alt < target_alt:
            # Let it gain some speed before pitching up
            if current_speed > 4.0:
                # Smoothly ramp up the elevator by 0.01 per step, capping at 0.12
                current_elevator = min(0.12, current_elevator + 0.01)
            else:
                current_elevator = 0.0 # Stay flat and accelerate
        else:
            current_elevator = 0.0
            throttle = 0.6 # Cruise

        action = np.array([aileron, current_elevator, throttle, rudder], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i} | Speed: {current_speed:.2f} m/s | Alt: {current_alt:.2f} m | Elev: {current_elevator:.3f}")

        if terminated:
            print("Crashed! Resetting...")
            obs, info = env.reset()

    print("Takeoff Test Complete. Did it fly?")
    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()