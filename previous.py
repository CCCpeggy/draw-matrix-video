import os
import cv2
import random
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Matrix:
    def __init__(self, width, height, padding, boundary_type='reflect'):
        self.padding = padding
        self.width = width
        self.height = height
        self.matrix = np.ones((width + padding * 2, height + padding * 2)) * 255
        self.boundary_type = boundary_type
        
    def set_value(self, i, j, value):
        self.matrix[i + self.padding, j + self.padding] = value
    
    def check_boundary(self, i, j):
        return i >= 0 and i < self.width and j >= 0 and j < self.height
        
    def get_value(self, i, j):
        if self.check_boundary(i, j):
            return self.matrix[i + self.padding, j + self.padding]
        else:
            if self.boundary_type == 'zero':
                return 0
            elif self.boundary_type == 'repeat':
                return self.get_value(i % self.width, j % self.height)
            elif self.boundary_type == 'reflect':
                # print(i, j, self.width - (i % self.width) - 1, self.height - (j % self.height))
                ii = i if i >= 0 and i < self.width else self.width - (i % self.width) - 1
                jj = j if j >= 0 and j < self.height else self.height - (j % self.height) - 1
                return self.get_value(ii, jj)
            return -1
        
    def do_function(self, i, j, function):
        self.matrix[i + self.padding, j + self.padding] = function(i, j)
    
    def select_value(self, ax, i, j, kernel):
        seleted_boundary_color = "#F7682D"
        kernel_boundary_color = "#E19133"
        kernel_text_color = "#F7BE2D"
        kernel_index = []
        for k in range(-(kernel // 2), (kernel - 1) // 2 + 1):
            for l in range(-(kernel // 2), (kernel - 1) // 2 + 1):
                position = (j-0.5+k+self.padding, i-0.5+l+self.padding)
                rect = Rectangle(position, 1, 1, fill=False, color=kernel_boundary_color, linewidth=3)
                ax.add_patch(rect)
                
                ax.text(j + l + self.padding, i + k + self.padding, int(self.get_value(i + k, j + l)), ha="center", va="center", color=kernel_text_color, fontsize=18)
                kernel_index.append((i + k + self.padding, j + l + self.padding))
        position = (j-0.5+self.padding, i-0.5+self.padding)
        rect = Rectangle(position, 1, 1, fill=False, color=seleted_boundary_color, linewidth=3)
        ax.add_patch(rect)
        return kernel_index
    
    def set_part_boundary(self, i, j, kernel, inverse=False):
        for k in range(-(kernel // 2), (kernel - 1) // 2 + 1):
            for l in range(-(kernel // 2), (kernel - 1) // 2 + 1):
                if not self.check_boundary(i + k, j + l):
                    matrix_before.set_value(i + k, j + l, matrix_before.get_value(i + k, j + l) if not inverse else 255)
                
    def draw(self, filename, selected_i=-1, selected_j=-1, selected_kernel=0):
        self.set_part_boundary(selected_i, selected_j, selected_kernel)
        # 繪製當前的矩陣
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.matrix, cmap='gray')
        ax.axis('off')  # 不顯示座標軸

        kernel_index = []
        if selected_kernel > 0:
            kernel_index = self.select_value(ax, selected_i, selected_j, selected_kernel)
        # 在每個單元格中添加數字
        for k in range(self.padding, width + self.padding):
            for l in range(self.padding, height + self.padding):
                if (k, l) not in kernel_index:
                    color = 'black' if self.matrix[k, l] > 127 else 'white'
                    ax.text(l, k, int(self.matrix[k, l]), ha="center", va="center", color=color, fontsize=18)
        plt.savefig(filename)
        plt.close()
        self.set_part_boundary(selected_i, selected_j, selected_kernel, True)
        
class FileNamer:
    def __init__(self):
        self.count = 0
    def get_name(self):
        filename = f'image_{self.count}.png'
        self.count += 1
        return filename
    def delete(self):
        for i in range(self.count):
            os.remove(f'image_{i}.png')


def function(matrix_before, matrix_after, func_name):
    if func_name == 'avg':
        def sub_function(i, j):
            sum = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    sum += matrix_before.get_value(i + k, j + l)
            return sum / 9
    elif func_name == 'draw':
        def sub_function(filename):
            img1, img2 = cv2.imread("tmp_1.png"), cv2.imread("tmp_2.png")
            # logo = cv2.imread("logo.png")
            h, w = img1.shape[:2]
            keep_h, keep_w = 410, 400
            crop_h, crop_w = (h - keep_h) // 2, (w - keep_w) // 2
            img1 = img1[crop_h:crop_h+keep_h, crop_w:crop_w+keep_w+50]
            img2 = img2[crop_h:crop_h+keep_h, crop_w+50:crop_w+keep_w]
            img = cv2.hconcat([img1, img2])
            # img = cv2.vconcat([img, logo])
            cv2.imwrite(filename, img)
    return sub_function
if __name__ == '__main__':
    width, height, kernel, padding = 5, 5, 3, 3
    matrix_before = Matrix(width, height, padding=padding)
    matrix_after = Matrix(width, height, padding=padding)
    filenamer = FileNamer()

    # 圖片列表，用於保存每個階段的圖片
    image_list = []
    for i in range(width):
        for j in range(height):
            value = int(random.random() * 255)
            matrix_before.set_value(i, j, value)
            matrix_after.set_value(i, j, value)

    matrix_before.draw("tmp_1.png")
    matrix_after.draw("tmp_2.png")
    for k in range(5):
        function(matrix_before, matrix_after, 'draw')(filenamer.get_name())
    for i in range(width):
        for j in range(height):
            matrix_after.do_function(i, j, function(matrix_before, matrix_after, 'avg'))
            # 保存圖片
            matrix_before.draw("tmp_1.png", i, j, kernel)
            matrix_after.draw("tmp_2.png", i, j, 1)
            for k in range(10):
                function(matrix_before, matrix_after, 'draw')(filenamer.get_name())
    matrix_before.draw("tmp_1.png")
    matrix_after.draw("tmp_2.png")
    for k in range(5):
        function(matrix_before, matrix_after, 'draw')(filenamer.get_name())
    
    # 使用ffmpeg來創建gif
    subprocess.call(['ffmpeg', '-i', 'image_%d.png', '-r', '20', '-vcodec', 'libx264', '-crf', '25', '-y', 'output.mp4'])
    # subprocess.call(['ffmpeg', '-i', 'image_%d.png', '-r', '20', 'output.gif'])

    # 最後刪除所有暫存的圖片
    filenamer.delete()
    os.remove(f'tmp_1.png')
    os.remove(f'tmp_2.png')
