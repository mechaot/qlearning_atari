# Install


* Install miniconda and add let installer add everything to path
* Install swig (comes now with environment.yml)


```bash
git clone https://github.com/mechaot/qlearning_atari
conda env create -n qlearning -f environment.yml

# to run
conda activate qlearning
python dqn_main.py
```


# Howto Run

Select Agent by modifying import and instance in `dqn_main.py`. Maybe adjust number of games `n_games`.

Run `python dqn_main.py`

Expect this to take several hours with pytoch+CUDA, several days on pytorch without CUDA.

# Note

We pin old python and old libs because newer versions are cumbersome/do not work: they lack the roms due to increasing copyright concerns.


# Ideas

Adopt other games without gym env:

python mss -> fast screenshots for other games
opencv -> template matching for extraction of scores/game over
pyautogui -> emulate user input and slow (?) screenshots
pynput -> emulate mouse/keyboard