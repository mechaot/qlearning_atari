# Install


* Install miniconda and add let installer add everything to path
* Install swig
  * Windows: Download and unzip swigwin 4.0.2, specify in bash snippet below
  * Linux: `sudo apt-get install swig`

```bash
export PATH=$PATH:./swigwin-4.0.2 # Win only

conda env create -n qlearning -f environment.yml
pip install -r requirements.txt

git clone https://github.com/mechaot/qlearning_atari
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