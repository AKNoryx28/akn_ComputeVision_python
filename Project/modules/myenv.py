from time import time
from cv2 import putText,  FONT_HERSHEY_PLAIN


class MyEnvironment:
    def __init__(self):
        # Colors
        #####################
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_LIME = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_YELLOW = (255, 255, 0)
        self.COLOR_CYAN = (0, 255, 255)
        self.COLOR_MAGENTA = (255, 0, 255)
        self.COLOR_SILVER = (192, 192, 192)
        self.COLOR_GRAY = (128, 128, 128)
        self.COLOR_MAROON = (128, 0, 0)
        self.COLOR_OLIVE = (128, 128, 0)
        self.COLOR_GREEN = (0, 128, 0)
        self.COLOR_PURPLE = (128, 0, 128)
        self.COLOR_TEA = (0, 128, 128)
        self.COLOR_NAVY = (0, 0, 128)
        #####################

    def DisplayFps(self, image, pos_wh=(10, 10), font_scale=1, color=(0, 0, 0), thickness=1, previous_time=0):
        current_time = time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        putText(image, f'fps: {int(fps)}', pos_wh,
                FONT_HERSHEY_PLAIN, font_scale, color, thickness)
        return previous_time

    def COLOR_WHITE(self):
        return self.COLOR_WHITE

    def COLOR_BLACK(self):
        return self.COLOR_BLACK

    def COLOR_BLUE(self):
        return self.COLOR_BLUE

    def COLOR_LIME(self):
        return self.COLOR_LIME

    def COLOR_RED(self):
        return self.COLOR_RED

    def COLOR_YELLOW(self):
        return self.COLOR_YELLOW

    def COLOR_CYAN(self):
        return self.COLOR_CYAN

    def COLOR_MAGENTA(self):
        return self.COLOR_MAGENTA

    def COLOR_SILVER(self):
        return self.COLOR_SILVER

    def COLOR_GRAY(self):
        return self.COLOR_GRAY

    def COLOR_MAROON(self):
        return self.COLOR_MAROON

    def COLOR_OLIVE(self):
        return self.COLOR_OLIVE

    def COLOR_GREEN(self):
        return self.COLOR_GREEN

    def COLOR_PURPLE(self):
        return self.COLOR_PURPLE

    def COLOR_TEA(self):
        return self.COLOR_TEA

    def COLOR_NAVY(self):
        return self.COLOR_NAVY
