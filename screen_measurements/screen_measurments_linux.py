#!/usr/bin/env python3
import subprocess
import re


def get_screen_size():
    '''
    find resolution and measutments in mm of the screen
    '''
    xrabdr_out = subprocess.check_output(["xrandr"]).decode("utf-8")
    xrandr_lines = xrabdr_out.strip().splitlines()
    for line in xrandr_lines:
        if " connected" in line:
            line = line.split()
            width_mm = int(line[-3].replace("mm", ""))
            height_mm = int(line[-1].replace("mm", ""))
            width_pixel = int(re.findall('[0-9][0-9]*', line[3])[0])
            height_pixel = int(re.findall('[0-9][0-9]*', line[3])[1])
    return width_mm, height_mm, width_pixel, height_pixel

if __name__=='__main__':
    print(('Screen measurments:\n'
          'width in mm:\t\t {} mm\n'
          'height in mm:\t\t {} mm\n'
          'width in pixels:\t {} pixels\n'
          'height in pixels:\t {} pixels\n').format(*get_screen_size()))
