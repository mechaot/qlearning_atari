# Install

python 3.7

download swigwin
```bash
export PATH=$PATH:/d/workspace/qlearning/swigwin-4.0.2
```

conda env create -n qlearning -f environment.yml

git clone https://github.com/openai/gym
cd gym 
pip install numpy scipy seaborn matplotlib pylint rope torch opencv-contrib-python

pip install atari-py ale-py gym[atari,accept-rom-license]


# Get ROMS

git clone https://github.com/openai/atari-py
cd atari-py
git checkout 77912fe3670492c2cf9952bd4a05e0383bd18f64
python -m atari_py.import_roms roms/

# Ideas

python mss -> fast screenshots
pyautogui -> emulate user input and slow (?) screenshots
pynput -> emulate mouse/keyboard