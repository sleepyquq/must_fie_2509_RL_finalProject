import os
import gymnasium as gym
import numpy as np
import imageio
import cv2
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# ==========================================
#               配置区域
# ==========================================
LOG_DIR = "./bipedal_sac_logs"
MODEL_DIR = os.path.join(LOG_DIR, "models")
VIDEO_DIR = os.path.join(LOG_DIR, "final_videos")
BEST_MODEL_DIR = os.path.join(LOG_DIR, "best_model")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

ENV_ID = "BipedalWalker-v3"
# SAC 效率很高，50万步通常就能达到 PPO 300万步的效果
# 如果你想追求极致稳定，可以设为 1000000
TOTAL_TIMESTEPS = 500000  
N_ENVS = 1 # SAC 是 Off-policy 算法，通常使用单线程环境效果最好，不用多核并行

print(f"检测到 CPU，正在启动 SAC 算法")
print(f"目标：实现双腿行走，直冲 300+ 分")
print("=" * 50)

# ==========================================
#             主训练流程
# ==========================================
def main():
    # 1. 创建环境
    # SAC 通常不需要并行环境，单进程即可
    env = make_vec_env(ENV_ID, n_envs=N_ENVS)
    # 依然需要归一化，这是物理环境的标配
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. 评估环境 (裁判)
    eval_env = make_vec_env(ENV_ID, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. 回调函数 (保存最佳模型)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=BEST_MODEL_DIR,
        eval_freq=10000,       # 每 1万步考一次
        n_eval_episodes=5,     # 每次考 5 局
        deterministic=True,
        verbose=1
    )

    # 4. 定义 SAC 模型 (针对 BipedalWalker 的黄金参数)
    # 来源：Stable Baselines3 RL Zoo 最佳实践
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device='cpu',
        batch_size=256,
        learning_rate=7.3e-4,  # SAC 的学习率通常比 PPO 大
        buffer_size=300000,    # 经验回放池
        learning_starts=10000, # 先随机乱动 1万步，收集数据
        train_freq=1,          # 每步都训练
        gradient_steps=1,
        ent_coef='auto',       # <--- 核心！自动调整探索欲望，绝不跪地！
        gamma=0.99,
        tau=0.01,
        policy_kwargs=dict(net_arch=[256, 256]), # 大脑容量保持 256
    )

    print("🚀 开始 SAC 训练...")
    print("提示：SAC 的 FPS 会比 PPO 慢，但它学的非常快！请耐心等待 50万步。")
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    
    print("训练完成。")
    
    # 保存最终模型
    model.save(os.path.join(MODEL_DIR, "final_model"))
    env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    
    env.close()
    eval_env.close()
    
    # 自动录像
    record_video(os.path.join(MODEL_DIR, "vec_normalize.pkl"))

# ==========================================
#             录像流程
# ==========================================
def record_video(stats_path):
    print("\n🎬 开始录制最终成果 (带步数与得分显示)...")
    
    best_model_path = os.path.join(BEST_MODEL_DIR, "best_model.zip")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(MODEL_DIR, "final_model.zip")
    
    print(f"加载模型: {best_model_path}")

    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = DummyVecEnv([lambda: env])
    
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    model = SAC.load(best_model_path, device='cpu')

    top_records = []
    
    print("-" * 30)
    # 测试 10 局
    for i in range(1, 11):
        obs = env.reset()
        frames = []
        total_reward = 0
        step_counter = 0 # 步数计数器
        
        while True:
            # 1. 获取原始画面
            frame = env.render()
            
            # 2. 转换为 OpenCV 可编辑格式 (复制一份，防止修改原始数据报错)
            # Gym 返回的是 RGB，OpenCV 也是处理数组，可以直接操作
            frame = np.array(frame, dtype=np.uint8)
            
            # 3. 准备文字内容
            step_counter += 1
            info_text = f"Step: {step_counter} | Score: {total_reward:.2f}"
            
            # 4. 在画面上写字 (带黑色描边，防止白色背景看不清)
            # 参数: 图片, 文字, 坐标(x,y), 字体, 大小, 颜色(RGB), 粗细
            # 先画黑色轮廓
            cv2.putText(frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            # 再画白色文字
            cv2.putText(frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            frames.append(frame)
            
            # 预测与执行
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            
            if dones[0]: break
        
        status = "👑 完美" if total_reward > 300 else ("✅ 优秀" if total_reward > 250 else "❌ 一般")
        print(f"测试 {i}/10: {total_reward:.2f} [{status}]")

        if len(top_records) < 3:
            top_records.append((total_reward, frames))
            top_records.sort(key=lambda x: x[0], reverse=True)
        elif total_reward > top_records[-1][0]:
            top_records.pop()
            top_records.append((total_reward, frames))
            top_records.sort(key=lambda x: x[0], reverse=True)

    print("正在保存前 3 名的视频...")
    for rank, (score, frames) in enumerate(top_records):
        filename = os.path.join(VIDEO_DIR, f"sac_rank{rank+1}_score_{score:.2f}.mp4")
        imageio.mimsave(filename, frames, fps=50)
        print(f"已保存: {filename}")
    
    env.close()


if __name__ == "__main__":
    main()