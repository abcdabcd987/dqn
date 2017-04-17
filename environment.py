import numpy as np
import threading
from PIL import Image
from ale_python_interface import ALEInterface

class AtariGame(object):
    _lock = threading.Lock()

    def __init__(self, romfile, random_seed, frame_skip):
        with AtariGame._lock:
            self.ale = ALEInterface()
            self.ale.setInt('random_seed', random_seed)
            self.ale.setInt('frame_skip', frame_skip)
            self.ale.setBool('color_averaging', True)
            self.hide_screen()
            self.ale.loadROM(romfile)
            self.action_space = self.ale.getLegalActionSet()

    def show_screen(self):
        if sys.platform == 'darwin':
            if '_pygame_inited' not in self:
                import pygame
                pygame.init()
                self._pygame_inited = True
            self.ale.setBool('sound', False)
        elif sys.platform.startswith('linux'):
            self.ale.setBool('sound', True)
        self.ale.setBool('display_screen', True)

    def hide_screen(self):
        self.ale.setBool('sound', False)
        self.ale.setBool('display_screen', False)

    def get_action_space(self):
        return self.action_space

    def get_image_shape(self):
        w, h = self.ale.getScreenDims()
        return w, h

    def reset(self, noop=0):
        self.ale.reset_game()
        for _ in xrange(noop):
            self.ale.act(0)

    def is_game_over(self):
        return self.ale.game_over()

    def act(self, action):
        '''return reward'''
        act = self.action_space[action]
        return self.ale.act(act)

    def get_image(self):
        '''shape: (height, width, 3)'''
        return self.ale.getScreenRGB()

    @staticmethod
    def preprocess(arr):
        img = Image.fromarray(arr)
        img = img.convert('L')
        img = img.resize((84, 84), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)

    def get_processed_image(self):
        return AtariGame.preprocess(self.get_image())