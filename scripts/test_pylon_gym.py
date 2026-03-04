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
    print(f"Initiating Takeoff & Hold at {target_alt}m (Press Ctrl+C to stop)...")

    current_elevator = -0.02 
    step_count = 0

    try:
        # Loop forever until Ctrl+C
        while rclpy.ok():
            current_alt = obs[2]
            current_speed = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)

            aileron = 0.0
            rudder = 0.0
            
            # 1. Proportional Error Calculation
            alt_error = target_alt - current_alt
            
            # 2. Flight Phase Logic
            if current_speed < 4.0:
                # Ground roll: tail down, full power
                target_elevator = -0.02
                throttle = 1.0
            elif current_speed >= 4.0 and current_alt < 2.0:
                # Initial climb: pitch up, full power
                target_elevator = 0.12
                throttle = 1.0
            else:
                # Airborne: Proportional control for altitude
                # Gain is 0.05. Clip max/min throw so it doesn't loop or dive.
                target_elevator = np.clip(alt_error * 0.05, -0.10, 0.15)
                
                # Throttle management based on error
                if alt_error > 1.0:
                    throttle = 0.8  # Need to climb
                elif alt_error < -1.0:
                    throttle = 0.3  # Need to descend
                else:
                    throttle = 0.6  # Cruise

            # 3. Smooth the actuator input (max movement of 0.01 per step)
            current_elevator += np.clip(target_elevator - current_elevator, -0.01, 0.01)

            action = np.array([aileron, current_elevator, throttle, rudder], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step_count % 10 == 0:
                print(f"Step {step_count} | Speed: {current_speed:.2f} m/s | Alt: {current_alt:.2f} m | Elev: {current_elevator:.3f} | Thr: {throttle:.2f}")

            if terminated:
                print("Crashed! Resetting...")
                obs, info = env.reset()
                current_elevator = -0.02

            step_count += 1

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down test script...")
    finally:
        env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()