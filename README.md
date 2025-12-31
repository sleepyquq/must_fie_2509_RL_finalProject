# BipedalWalker Reinforcement Learning Project

## Project Description
Train BipedalWalker-v3 using the SAC algorithm, analyze performance, and generate demonstration videos.

## Environment Dependencies
- Python 3.8 - 3.10 (Recommended)
- Windows, Linux, or macOS

## Installation Steps
1. Open terminal or command prompt.
2. Create virtual environment:
   python -m venv .venv
3. Activate virtual environment:
   - Windows: .venv\Scripts\activate
   - Linux/Mac: source .venv/bin/activate
4. Install dependencies:
   pip install -r requirements.txt

   use gpu pip this:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120  ##cheack your cuda version

   **Important Note on Hardware Selection**

We strongly recommend using CPU for training.
Experimental Verification: Based on our extensive testing, training on the CPU consistently yields stable and reproducible results, achieving scores of 300+ with a natural walking gait.
Potential Issues with GPU: We have observed that training on GPU can sometimes lead to suboptimal performance (e.g., the agent getting stuck in local optima such as "kneeling" or "hopping," failing to converge to high scores).
Reasoning: This discrepancy is likely due to Hardware Determinism and Floating-Point Precision:
Physics Sensitivity: The Box2D physics engine is highly sensitive to minor perturbations. The difference in floating-point arithmetic between CPU and GPU (CUDA) can introduce slight rounding errors. In chaotic physical environments, these errors amplify, potentially steering the agent toward conservative strategies (like kneeling) during early exploration.
Network Overhead: Given the relatively small network size ([256, 256]), the overhead of data transfer between CPU and GPU outweighs the parallel computation benefits. Thus, CPU training proves to be more efficient and stable for this specific task.

## Run Project
Run the following commands based on your needs:

- **Start Training:**
  python main.py --train

- **Run Live Demo (requires trained model):**
  python main.py --demo

- **Record Videos:**
  python main.py --video

- **Generate Data Plots:**
  python main.py --plot

## Output
After the program finishes, the `.\bipedal_sac_logs` directory will contain:
- **final_videos**: The top 3 highest-scoring videos.
- **plots**: Performance analysis charts (Learning Curve, Stability).
- **training_result_log.txt**: Detailed training log.



