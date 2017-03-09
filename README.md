# dqn

```bash
###### atari
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake # for ubuntu
brew install sdl sdl_gfx sdl_image # for mac
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment
mkdir build && cd build
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
make -j 4
pip install .

###### dqn
git clone https://github.com/abcdabcd987/dqn.git
cd dqn
mkdir rom
wget https://atariage.com/2600/roms/Breakout.zip
unzip Breakout.zip
mv Breakout.bin rom

###### run
python main.py
```

## reference

* <https://github.com/songrotek/DQN-Atari-Tensorflow>
* <https://github.com/devsisters/DQN-tensorflow>
