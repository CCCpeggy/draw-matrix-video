import os
import cv2
import shutil
import random
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class MatrixBoundaryManager:
    ZERO = 0
    REPEAT = 1
    REFLECT = 2
    def __init__(self, matrix, boundary_type="zero"):
        self.matrix = matrix
        self.boundary_type = boundary_type
    
    def __call__(self, i, j):
        if self.boundary_type == MatrixBoundaryManager.ZERO:
            return 0
        elif self.boundary_type == MatrixBoundaryManager.REPEAT:
            return self.matrix(i % self.matrix.size, j % self.matrix.size)
        elif self.boundary_type == MatrixBoundaryManager.REFLECT:
            ii = i if i >= 0 and i < self.matrix.size else self.matrix.size - (i % self.matrix.size) - 1
            jj = j if j >= 0 and j < self.matrix.size else self.matrix.size - (j % self.matrix.size) - 1
            return self.matrix(ii, jj)
        return -1

class KernelManager:
    def __init__(self, kernel):
        self.kernel = kernel
        self.size = self.kernel.shape[0]
        self.drawer = MatrixDrawer(self, 4)
    
    def get_index(self, i=0, j=0):
        shift = -(self.size // 2)
        for k in range(0, self.size):
            for l in range(0, self.size):
                yield k, l, i + k + shift, j + l + shift
    
    @staticmethod
    def get_kernel_index(size, i=0, j=0):
        shift = -(size // 2)
        for k in range(0, size):
            for l in range(0, size):
                yield k, l, i + k + shift, j + l + shift
    
    def normalize(self):
        self.kernel = self.kernel.astype(np.float32)
        self.kernel /= self.kernel.sum()
        
    def __call__(self, i, j):
        return self.kernel[i, j]

class MatrixManager:
    def __init__(self, size, padding, boundary_type=MatrixBoundaryManager.ZERO):
        self.matrix = np.zeros((size, size))
        self.size = size
        self.boundary_manager = MatrixBoundaryManager(self, boundary_type)
        self.drawer = MatrixDrawer(self, padding)
    
    def random(self):
        for i in range(self.size):
            for j in range(self.size):
                value = int(random.random() * 255)
                self.set_value(i, j, value)

    def check_boundary(self, i, j):
        return i >= 0 and i < self.size and j >= 0 and j < self.size

    def set_value(self, i, j, value):
        if self.check_boundary(i, j):
            self.matrix[i, j] = value

    def __call__(self, i, j):
        if self.check_boundary(i, j):
            return self.matrix[i, j]
        return self.boundary_manager(i, j)

    def convolve(self, i, j, kernel):
        sum = 0
        for ki, kj, mi, mj in kernel.get_index(i, j):
            sum += self(mi, mj) * kernel(ki, kj)
        return sum
    
class TmpFileNamer:
    def __init__(self, idx=0):
        self.count = 0
        self.tmp_files = []
        self.tmp_folder = f"tmp_files_{idx}"
        if not os.path.exists(self.tmp_folder):
            os.mkdir(self.tmp_folder)

    def get_name(self):
        filename = f'image_{self.count}.png'
        self.count += 1
        if filename not in self.tmp_files:
            self.tmp_files.append(filename)
        return os.path.join(self.tmp_folder, filename)

    def get_format(self):
        return os.path.join(self.tmp_folder, f'image_%d.png')

    def get_tmp_name(self, idx=0):
        filename = f'image_tmp_{idx}.png'
        if filename not in self.tmp_files:
            self.tmp_files.append(filename)
        return os.path.join(self.tmp_folder, filename)

    def delete(self):
        for filename in self.tmp_files:
            os.remove(os.path.join(self.tmp_folder, filename))
        os.rmdir(self.tmp_folder)
        self.tmp_files = []

class MatrixDrawer:
    seleted_boundary_color = "#F7682D"
    kernel_boundary_color = "#E19133"
    kernel_text_color = "#F7BE2D"
    def __init__(self, matrix, padding):
        self.matrix = matrix
        self.padding = padding
        
    def draw_rect(self, ax, i, j, color="black"):
        position = (j-0.5+self.padding, i-0.5+self.padding)
        rect = Rectangle(position, 1, 1, fill=False, color=color, linewidth=3)
        ax.add_patch(rect)
          
    def draw_value(self, ax, i, j, color=None):
        if color is None:
            color = 'black' if self.matrix(i, j) > 127 else 'white'
        ax.text(j + self.padding, i + self.padding, int(self.matrix(i, j)), ha="center", va="center", color=color, fontsize=18)
        
    def __call__(self, filename, selected_i=-1, selected_j=-1, kernel_size=0):
        if isinstance(self.matrix, KernelManager):
            self.draw_kernel(filename)
            return
        matrix = self.matrix.matrix * 1
        matrix = np.pad(matrix, (self.padding,), 'constant', constant_values=(255))
        # pad_fst_start, matrix_start, pad_snd_start, total_len = 0, self.padding, self.padding + matrix.size, self.padding * 2 + matrix.size
        
        if kernel_size > 0:
            for ki, kj, mi, mj in KernelManager.get_kernel_index(kernel_size, selected_i, selected_j):
                if not self.matrix.check_boundary(mi, mj):
                    matrix[mi + self.padding, mj + self.padding] = self.matrix(mi, mj)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(matrix, cmap='gray')
        ax.axis('off')  # 不顯示座標軸
        
        # 繪製當前的 kernel
        kernel_index = []
        if kernel_size > 0:
            for ki, kj, mi, mj in KernelManager.get_kernel_index(kernel_size, selected_i, selected_j):
                if not (mi == selected_i and mj == selected_j):
                    self.draw_rect(ax, mi, mj, MatrixDrawer.kernel_boundary_color)
                self.draw_value(ax, mi, mj, MatrixDrawer.kernel_text_color)
                kernel_index.append((mi, mj))
            self.draw_rect(ax, selected_i, selected_j, MatrixDrawer.seleted_boundary_color)

        # 在每個單元格中添加數字
        for i in range(self.matrix.size):
            for j in range(self.matrix.size):
                if (i, j) not in kernel_index:
                    self.draw_value(ax, i, j)
                    
        plt.savefig(filename)
        plt.close()
        
    def draw_kernel(self, filename):
        matrix = np.ones((self.matrix.size, self.matrix.size)) * 255
        matrix = np.pad(matrix, (self.padding - 1,), 'constant', constant_values=(255))
        matrix = np.pad(matrix, (1,), 'constant', constant_values=(0))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(matrix, cmap='gray')
        ax.axis('off')  # 不顯示座標軸
        
        # 繪製當前的 kernel
        for ki, kj, mi, mj in self.matrix.get_index():
            self.draw_rect(ax, ki, kj, "black")
            self.draw_value(ax, ki, kj, "black")
                    
        plt.savefig(filename)
        plt.close()

class MatrixVideoGenerator:
    def __init__(self, matrix_size, kernel, change_based=False):
        self.matrix_size=matrix_size
        self.padding=kernel.shape[0] // 2 + 1
        self.change_based = change_based

        self.matrix_before = MatrixManager(matrix_size, self.padding, MatrixBoundaryManager.ZERO)
        self.matrix_before.random()
        self.matrix_after = MatrixManager(matrix_size, self.padding, MatrixBoundaryManager.ZERO)
        self.matrix_after.matrix = self.matrix_before.matrix * 1
        self.kernel=KernelManager(kernel)
        self.kernel.normalize()
        self.file_namer_1 = TmpFileNamer(1)
        self.file_namer_2 = TmpFileNamer(2)
 
    def __call__(self):
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if self.change_based:
                    self.matrix_before.set_value(i, j, self.matrix_before.convolve(i, j, self.kernel))
                    self.matrix_before.drawer(self.file_namer_1.get_tmp_name(), i, j, self.kernel.size)
                else:
                    self.matrix_after.set_value(i, j, self.matrix_before.convolve(i, j, self.kernel))
                    self.matrix_before.drawer(self.file_namer_1.get_tmp_name(), i, j, self.kernel.size)
                    self.matrix_after.drawer(self.file_namer_2.get_tmp_name(), i, j, 1)
                for _ in range(10):
                    shutil.copyfile(self.file_namer_1.get_tmp_name(), self.file_namer_1.get_name())
                    if not self.change_based:
                        shutil.copyfile(self.file_namer_2.get_tmp_name(), self.file_namer_2.get_name())
        self.kernel.drawer("kernel.png")
        
        subprocess.call(['ffmpeg', '-i', self.file_namer_1.get_format(), '-r', '20', '-vcodec', 'libx264', '-crf', '25', '-y', 'matrix1.mp4'])
        if not self.change_based:
            subprocess.call(['ffmpeg', '-i', self.file_namer_2.get_format(), '-r', '20', '-vcodec', 'libx264', '-crf', '25', '-y', 'matrix2.mp4'])
        self.file_namer_1.delete()
        self.file_namer_2.delete()
                
if __name__ == "__main__":
    MatrixVideoGenerator(5, np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]]), change_based=False)()
