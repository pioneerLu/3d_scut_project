import numpy as np 
import cv2 as cv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt
import time
import os

class TwoInputBox(tk.simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="请输入您的肩宽:").grid(row=0)
        tk.Label(master, text="请输入您的身高:").grid(row=1)
        

        self.entry1 = tk.Entry(master)
        self.entry2 = tk.Entry(master)

        self.entry1.grid(row=0, column=1)
        self.entry2.grid(row=1, column=1)

    def apply(self):
        self.value1 = self.entry1.get()
        self.value2 = self.entry2.get()



window_name = "Distance"
initial_threshold = 1000  # 初始距离阈值(mm)
min_threshold = 400      # 最小距离阈值(mm)
auto_threshold_counter = 0
auto_threshold_interval = 1    # 自动阈值flag
square_size_threshold = 50     # 矩形框阈值

def set_thre1(thre):
    global initial_threshold
    initial_threshold = int(thre)

def find_square(image, m, n):
    # 在图像中查找是否存在特定大小的矩形框，框内的值全为1
    for i in range(len(image) - m):
        for j in range(len(image[0]) - n):
            if np.all(image[i:i+m, j:j+n] == 1):
                return False
    return True

def depth_to_pointcloud(depth_map):
    """将深度图转换为点云"""
    h, w = depth_map.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_map
    return X, Y, Z

pointcloud_image_counter = 1


def update_pointcloud_plot(X, Y, Z):
    """生成点云图并保存到文件"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=1)

    # 指定保存路径
    save_path = "./pointcloud"
    os.makedirs(save_path, exist_ok=True)

    # 创建文件名，包括编号
    filename = os.path.join(save_path, f"pointcloud_{pointcloud_image_counter}.png")

    # 保存图像
    plt.savefig(filename)

    # 关闭图形，释放资源
    plt.close(fig)


def main():
    global pointcloud_image_counter
    two_input_dialog = TwoInputBox(root, "信息输入")
    print("你输入的第一个值是:", two_input_dialog.value1)
    print("你输入的第二个值是:", two_input_dialog.value2)
    shoulder_width = float(two_input_dialog.value1)  # cm输入
    height = float(two_input_dialog.value2)  
    
    cv.namedWindow(window_name)
    cv.createTrackbar("dist1", window_name, initial_threshold, 1000, set_thre1)

    orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture!")
        exit(0)



    while cv.waitKey(1) < 0:
       

        last_update_time = 0
        update_interval = 1  # 点云图更新间隔（秒）
        if orbbec_cap.grab():
            ret_bgr, bgr_image = orbbec_cap.retrieve(flag=cv.CAP_OBSENSOR_BGR_IMAGE)
            ret_depth, depth_map = orbbec_cap.retrieve(flag=cv.CAP_OBSENSOR_DEPTH_MAP)
            barrier_mat = np.zeros(depth_map.shape)

            # 获取当前阈值
            current_threshold = cv.getTrackbarPos("dist1", window_name)
            # 计算 m 和 n
            k = 0.005  # 可以根据相机内参调整 k 和 b 的值
            b = 0
            f_D = k * current_threshold + b
            m = int(1.1 * height * f_D)
            n = int(1.1 * shoulder_width * f_D)
            print(str(m)+"   "+str(n))
            # print(current_threshold)
            for i in range(len(depth_map)):
                for j in range(len(depth_map[0])):
                    if depth_map[i][j] < current_threshold:
                        barrier_mat[i][j] = 0
                    else:
                        barrier_mat[i][j] = 1

            if ret_bgr and ret_depth:


                if ret_depth:
                    current_time = time.time()
                    if current_time - last_update_time > update_interval:
                        last_update_time = current_time
                        # 生成并更新点云图
                        X, Y, Z = depth_to_pointcloud(depth_map)
                        update_pointcloud_plot( X, Y, Z)
                        pointcloud_image_counter +=1
                color_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
                cv.imshow("Depth",  color_depth_map)

                cv.imshow('barrier', barrier_mat)

                # 显示当前阈值
                cv.putText(bgr_image, f"Threshold: {current_threshold}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.imshow(window_name, bgr_image)
                
                # 自动调节阈值
                global auto_threshold_counter
                auto_threshold_counter += 1

                if auto_threshold_counter >= auto_threshold_interval:
                    auto_threshold_counter = 0

                    # 判断是否存在特定大小的矩形框，框内的值全为1
                    if find_square(barrier_mat, m,n) and current_threshold >= min_threshold:
                        current_threshold -= 50  # 自动减小阈值
                        cv.setTrackbarPos("dist1", window_name, current_threshold)

                        if current_threshold <= min_threshold:
                            print("Turn Warning: Minimum threshold reached. Please turn to other directions")
                            cv.setTrackbarPos("dist1", window_name, initial_threshold)  # 重置为初始值
                    if current_threshold <= min_threshold:
                        print("Turn Warning: Minimum threshold reached. Please turn to other directions")
                        cv.setTrackbarPos("dist1", window_name, 1000)
                
        else:
            print("Fail to grab data from camera!")
    
    orbbec_cap.release()

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    main()
