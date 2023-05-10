#!/usr/bin/env python

import time

import torch
import numpy as np
import cv2

import rospy
import ros_numpy
from sensor_msgs.msg import Image

from maskrcnn_ros.srv import InstanceSeg, InstanceSegResponse
from segmentation import Segmentation


def seg_callback(req):
    global seg
    img: np.ndarray = ros_numpy.numpify(req.input)
    shape = img.shape
    rospy.loginfo(f"开始实例分割 {shape}")
    t_start = time.perf_counter()

    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) / 255
    img = np.clip(img - img.mean(), -1, 1)
    try:
        _, predict_classes, _, predict_masks = seg.predict(img)
    except Exception as e:
        rospy.logerr(f"Maskrcnn预测失败: {e}")
        res = InstanceSegResponse()
        res.seg = ros_numpy.msgify(Image, np.zeros_like(img,dtype=np.uint8), encoding="mono8")
        return res
    label_mask = np.zeros(shape[:2], dtype=np.uint8)
    num_instances = len(predict_masks)
    class_map=[0]*(num_instances+1)
    for i in range(num_instances):
        index = np.where(predict_masks[i] ==1)
        label_mask[index] = i + 1
        class_map[i+1]=predict_classes[i]

    t_end = time.perf_counter()
    rospy.loginfo(f"实例分割完成,检测到 {num_instances} 个实例,耗时 {(t_end - t_start)*1000:.2f}ms")

    res = InstanceSegResponse()
    res.seg = ros_numpy.msgify(Image, label_mask, encoding="mono8")
    return res

def pseudo_seg(req):
    img: np.ndarray = ros_numpy.numpify(req.input)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    color_num=4

    color_ranges=[
                 [np.array([-10, 40, 0]),np.array([10, 255, 255])],#red
                  [np.array([100, 40, 0]),np.array([120, 255, 255])],#blue
                  [np.array([20, 40, 0]),np.array([40, 255, 255])],#yellow
                  [np.array([45, 40, 0]),np.array([50, 255, 255])],#green
                  ]

    color_masks=[]
    for i in range(color_num):
        mask=cv2.inRange(hsv_img, color_ranges[i][0], color_ranges[i][1])
        color_masks.append(mask)

    output = np.zeros(img.shape[:-1],dtype=np.uint8) # 用于保存分割结果
    obj_img=np.ones_like(img,dtype=np.uint8)*255
    obj_class=[0]
    obj_id=1
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    for i in range(color_num):
        mask=color_masks[i]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1,iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2,iterations=1)
        mask = cv2.erode(mask, kernel2, iterations=1)
        num_objs,labels=cv2.connectedComponents(mask)
        print(num_objs)
        for j in range(num_objs-1):
            output[np.where(labels==j+1)]=obj_id
            obj_img[np.where(labels==j+1)]=img[np.where(labels==j+1)]
            print(f"第 {obj_id} 个实例属于第 {i+1} 种类别")
            obj_id+=1
            obj_class.append(i+1)

    print(f"实例分割完成,检测到 {obj_id-1} 个实例，属于 {len(set(obj_class))-1} 种类别")
    res = InstanceSegResponse()
    res.seg = ros_numpy.msgify(Image, output, encoding="mono8")
    res.classes=obj_class
    return res


if __name__ == "__main__":
    # model_path = "../models/model_39.pth"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # rospy.logdebug(f"加载实例分割网络 {model_path}")
    # seg = Segmentation(model_path, device)

    rospy.init_node("maskrcnn_ros_server")
    s = rospy.Service("/maskrcnn_ros_server/seg_instance", InstanceSeg, pseudo_seg)
    # s = rospy.Service("/maskrcnn_ros_server/seg_instance", InstanceSeg, seg_callback)
    rospy.loginfo("/maskrcnn_ros_server 节点就绪")
    rospy.spin()
