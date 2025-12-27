import os
import argparse
import gymnasium as gym
import numpy as np
import imageio
import cv2
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# ==========================================
#       Configuration & Hyperparameters
# ==========================================
LOG_DIR = "./bipedal_sac_logs"
MODEL_DIR = os.path.join(LOG_DIR, "models")
VIDEO_DIR = os.path.join(LOG_DIR, "final_videos")
BEST_MODEL_DIR = os.path.join(LOG_DIR, "best_model")
PLOT_DIR = os.path.join(LOG_DIR, "plots")
STATS_PATH = os.path.join(MODEL_DIR, "vec_normalize.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

ENV_ID = "BipedalWalker-v3"
TOTAL_TIMESTEPS = 500000
N_ENVS = 2  # Optimized for GPU throughput

device_str = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
#             Training Logic
# ==========================================
def train():
    print(f"[INFO] Device: {device_str.upper()} | Environment: {ENV_ID}")
    print("[INFO] Initializing training environments...")

    # 1. Training Environment
    env = make_vec_env(ENV_ID, n_envs=N_ENVS)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Evaluation Environment
    eval_env = make_vec_env(ENV_ID, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=BEST_MODEL_DIR,
        eval_freq=10000 // N_ENVS,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )

    # 4. Model Initialization
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device=device_str,
        batch_size=256,
        learning_rate=7.3e-4,
        buffer_size=300000,
        learning_starts=10000,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        gamma=0.99,
        tau=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    print("[INFO] Starting training phase...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    print("[INFO] Training complete.")
    
    # Save artifacts
    model.save(os.path.join(MODEL_DIR, "final_model"))
    env.save(STATS_PATH)
    
    env.close()
    eval_env.close()
    
    # Auto-generate plots after training
    generate_plots()

# ==========================================
#        Data Visualization & Logging
# ==========================================
def generate_plots():
    print("[INFO] Generating analysis plots and logs...")
    data_path = os.path.join(BEST_MODEL_DIR, "evaluations.npz")
    
    if not os.path.exists(data_path):
        print("[WARN] Evaluation data not found. Skipping plots.")
        return

    data = np.load(data_path)
    timesteps = data['timesteps']
    results = data['results']       
    ep_lengths = data['ep_lengths'] 

    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    mean_lengths = np.mean(ep_lengths, axis=1)

    # TXT Log
    txt_log_path = os.path.join(LOG_DIR, "training_result_log.txt")
    with open(txt_log_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"           SAC Training Log: {ENV_ID}\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Timestep':<12} | {'Mean Reward':<12} | {'Std Dev':<10} | {'Mean Length':<12}\n")
        f.write("-" * 60 + "\n")
        
        best_reward = -np.inf
        best_step = 0
        
        for i in range(len(timesteps)):
            t = timesteps[i]
            r = mean_rewards[i]
            std = std_rewards[i]
            l = mean_lengths[i]
            if r > best_reward:
                best_reward = r
                best_step = t
            f.write(f"{t:<12} | {r:<12.2f} | {std:<10.2f} | {l:<12.2f}\n")
            
        f.write("-" * 60 + "\n")
        f.write(f"Best Performance: Reward {best_reward:.2f} at Step {best_step}\n")
        f.write("=" * 60 + "\n")
    
    # Plot 1: Learning Curve
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label="Mean Reward", color='b', linewidth=2)
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, color='b', alpha=0.2, label="Std Dev")
    plt.axhline(y=300, color='r', linestyle='--', label="Solved Threshold (300)")
    plt.title("SAC Training Performance: Average Reward over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "learning_curve_reward.png"))
    plt.close()

    # Plot 2: Episode Length
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_lengths, label="Mean Episode Length", color='g', linewidth=2)
    plt.title("Agent Stability: Average Episode Length over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Steps per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "training_episode_length.png"))
    plt.close()

    # Plot 3: Stability Boxplot
    indices = np.linspace(0, len(timesteps)-1, 5, dtype=int)
    selected_data = [results[i] for i in indices]
    selected_labels = [f"{int(timesteps[i]/1000)}k" for i in indices]

    plt.figure(figsize=(10, 6))
    # Corrected argument 'labels' for matplotlib compatibility
    plt.boxplot(selected_data, labels=selected_labels, patch_artist=True)
    plt.title("Reward Distribution Analysis")
    plt.xlabel("Timesteps (k = 1000)")
    plt.ylabel("Reward Distribution")
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "reward_stability_boxplot.png"))
    plt.close()
    
    print(f"[INFO] Plots saved to {PLOT_DIR}")

# ==========================================
#           Video Recording
# ==========================================
def record_video(stats_path):
    print("[INFO] Recording top 3 performing episodes...")
    
    best_model_path = os.path.join(BEST_MODEL_DIR, "best_model.zip")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(MODEL_DIR, "final_model.zip")
    
    if not os.path.exists(stats_path):
        print(f"[ERROR] Stats file not found at {stats_path}. Train model first.")
        return

    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    model = SAC.load(best_model_path, device=device_str)
    top_records = []
    
    for i in range(1, 11):
        obs = env.reset()
        frames = []
        total_reward = 0
        step_counter = 0
        
        while True:
            frame = env.render()
            frame = np.array(frame, dtype=np.uint8)
            step_counter += 1
            
            info_text = f"Step: {step_counter} | Score: {total_reward:.2f}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(frame)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            
            if dones[0]: break
        
        status = "Solved" if total_reward > 300 else "Finished"
        print(f"Test Episode {i}/10: Score {total_reward:.2f} [{status}]")

        if len(top_records) < 3:
            top_records.append((total_reward, frames))
            top_records.sort(key=lambda x: x[0], reverse=True)
        elif total_reward > top_records[-1][0]:
            top_records.pop()
            top_records.append((total_reward, frames))
            top_records.sort(key=lambda x: x[0], reverse=True)

    for rank, (score, frames) in enumerate(top_records):
        filename = os.path.join(VIDEO_DIR, f"sac_rank{rank+1}_score_{score:.2f}.mp4")
        imageio.mimsave(filename, frames, fps=50)
        print(f"[INFO] Video saved: {filename}")
    
    env.close()

# ==========================================
#          Interactive Live Demo
# ==========================================
def run_live_demo(stats_path):
    print("\n[INFO] Starting Live Demo Window (Press 'q' to exit)...")
    
    best_model_path = os.path.join(BEST_MODEL_DIR, "best_model.zip")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(MODEL_DIR, "final_model.zip")

    if not os.path.exists(stats_path):
        print(f"[ERROR] Stats file not found at {stats_path}. Train model first.")
        return
    
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    model = SAC.load(best_model_path, device=device_str)

    for i in range(1, 6):
        obs = env.reset()
        total_reward = 0
        step_counter = 0
        current_step_reward = 0.0
        
        print(f"Demo Episode {i}/5 started...")

        while True:
            frame = env.render()
            img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            step_counter += 1
            
            reward_color = (0, 255, 0) if current_step_reward >= 0 else (0, 0, 255)
            text_color = (255, 255, 255)
            
            txt_step = f"Step: {step_counter}"
            txt_total = f"Total: {total_reward:.2f}"
            txt_instant = f"Instant: {current_step_reward:+.2f}"

            def draw_text(img, text, pos, color, scale=0.6):
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

            draw_text(img_bgr, txt_step, (10, 30), text_color)
            draw_text(img_bgr, txt_total, (10, 60), text_color)
            draw_text(img_bgr, txt_instant, (10, 90), reward_color)

            cv2.imshow("SAC BipedalWalker Agent", img_bgr)
            
            if cv2.waitKey(20) == ord('q'):
                env.close()
                cv2.destroyAllWindows()
                return

            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            current_step_reward = rewards[0]
            total_reward += current_step_reward
            
            if dones[0]:
                final_text = f"Result: Steps {step_counter} | Score {total_reward:.2f}"
                draw_text(img_bgr, final_text, (50, 200), (0, 255, 255), scale=0.8)
                draw_text(img_bgr, "Press any key to continue...", (50, 240), (255, 255, 255), scale=0.6)
                cv2.imshow("SAC BipedalWalker Agent", img_bgr)
                print(f"[INFO] Episode {i} finished. {final_text}")
                cv2.waitKey(0)
                break
                
    env.close()
    cv2.destroyAllWindows()

# ==========================================
#             Main Entry Point
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="SAC BipedalWalker-v3 Manager")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--plot", action="store_true", help="Generate analysis plots")
    parser.add_argument("--video", action="store_true", help="Record validation videos")
    parser.add_argument("--demo", action="store_true", help="Run real-time visualization")
    
    args = parser.parse_args()

    # If no arguments are provided, print help
    if not any(vars(args).values()):
        parser.print_help()
        return

    # Execute based on flags
    if args.train:
        train()
        
    if args.plot and not args.train: # If trained, plot is already called
        generate_plots()
        
    if args.video:
        record_video(STATS_PATH)
        
    if args.demo:
        run_live_demo(STATS_PATH)

if __name__ == "__main__":
    main()
