import numpy as np 
import cv2 as cv
import numpy as np
import cv2 as cv
import torch
import threading

model = torch.hub.load('E:\project\YOLO\yolov5', 'custom', 'yolov5x.pt', source='local')
model = model.cuda()




window_name = "Distance"
initial_threshold = 1000  # 初始距离阈值(mm)
min_threshold = 300      # 最小距离阈值(mm)
auto_threshold_counter = 0
auto_threshold_interval = 1    # 自动阈值flag
square_size_threshold = 50     # 矩形框阈值

def set_thre1(thre):
    global initial_threshold
    initial_threshold = int(thre)

def find_square(image, square_size=5):
    # 在图像中查找是否存在特定大小的矩形框，框内的值全为1
    for i in range(len(image) - square_size):
        for j in range(len(image[0]) - square_size):
            if np.all(image[i:i+square_size, j:j+square_size] == 1):
                return False
    return True


def detect_objects():
    orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture!")
        exit(0)
    while True:
        if orbbec_cap.grab():
            ret_bgr, bgr_image = orbbec_cap.retrieve(flag=cv.CAP_OBSENSOR_BGR_IMAGE)

            if ret_bgr:
                try:
                    # 在图像上运行Yolov5模型进行移动物体检测
                    detections = model(bgr_image)

                    # 处理检测结果
                    for detection in detections.pred[0]:
                        class_id, confidence, x1, y1, x2, y2 = detection.tolist()[:6]
                        class_name = model.names[int(class_id)]

                        # 如果检测到可移动物体（'car'、'bus'、'person'）并且置信度大于0.5，发出提示并绘制边界框
                        if class_name in ['car', 'bus', 'person'] and confidence > 0.5:
                            print("检测到了可移动物体！")
                            # 在图像上绘制边界框
                            cv.rectangle(bgr_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # 在窗口中显示带有检测结果的图像
                    cv.imshow(window_name, bgr_image)

                except Exception as e:
                    print(f"无移动障碍风险")
                    continue
            # 程序退出时释放资源
        orbbec_cap.release()
        cv.destroyAllWindows()

def process_depth_image():
    global auto_threshold_counter
    orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture!")
        exit(0)
    while True:
        
        if orbbec_cap.grab():
            ret_bgr, bgr_image = orbbec_cap.retrieve(flag=cv.CAP_OBSENSOR_BGR_IMAGE)
            ret_depth, depth_map = orbbec_cap.retrieve(flag=cv.CAP_OBSENSOR_DEPTH_MAP)
            barrier_mat = np.zeros(depth_map.shape)

            # 获取当前阈值
            current_threshold = cv.getTrackbarPos("dist1", window_name)
            for i in range(len(depth_map)):
                for j in range(len(depth_map[0])):
                    if depth_map[i][j] < current_threshold:
                        barrier_mat[i][j] = 0
                    else:
                        barrier_mat[i][j] = 1

            if ret_depth:
                color_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
                cv.imshow("Depth",  color_depth_map)

                cv.imshow('barrier', barrier_mat)

                # 显示当前阈值
                cv.putText(bgr_image, f"Threshold: {current_threshold}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv.imshow(window_name, bgr_image)

                # 自动调节阈值
                global auto_threshold_counter
                auto_threshold_counter += 1

                if auto_threshold_counter >= auto_threshold_interval:
                    auto_threshold_counter = 0

                    # 判断是否存在特定大小的矩形框，框内的值全为1
                    if find_square(barrier_mat, square_size=square_size_threshold) and current_threshold >= min_threshold:
                        current_threshold -= 50  # 自动减小阈值
                        cv.setTrackbarPos("dist1", window_name, current_threshold)

                    if current_threshold <= min_threshold:
                        print("Turn Warning: Minimum threshold reached. Please turn to other directions")
                        cv.setTrackbarPos("dist1", window_name, 1000)

            else:
                print("Fail to grab data from camera!")
            # 程序退出时释放资源
        orbbec_cap.release()
        cv.destroyAllWindows()
def main():
    
    
    


    depth_thread = threading.Thread(target=process_depth_image)
    detect_thread = threading.Thread(target=detect_objects)

    depth_thread.start()
    detect_thread.start()

    # 主线程等待两个子线程完成
    depth_thread.join()
    detect_thread.join()

    


if __name__ == '__main__':
    main()
