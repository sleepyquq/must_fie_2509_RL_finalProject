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
   -use gpu pip this
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120  ##cheack your cuda version 

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


