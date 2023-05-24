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
    save_inter_data=False
    img: np.ndarray = ros_numpy.numpify(req.input)
    t=int(time.time())
    if save_inter_data:
        np.save(f"../tests/img{t}.npy",img)
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
        res.seg = ros_numpy.msgify(
            Image, np.zeros_like(img, dtype=np.uint8), encoding="mono8"
        )
        return res
    label_mask = np.zeros(shape[:2], dtype=np.uint8)
    num_instances = len(predict_masks)
    class_map = []
    inst_id = 1
    for i in range(num_instances):
        if predict_classes[i] in [67, 84]:  # 跳过coco中容易误判的
            continue
        inst_mask = np.zeros_like(label_mask, dtype=bool)
        inst_mask[np.where(predict_masks[i] > 0.1)] = True
        index = np.logical_and(inst_mask, label_mask == 0)
        label_mask[index] = inst_id
        inst_id += 1
        class_map.append(predict_classes[i])
    if save_inter_data:
        np.save(f"../tests/mask{t}.npy",label_mask)

    t_end = time.perf_counter()
    rospy.loginfo(f"实例分割完成,检测到 {num_instances} 个实例{class_map},耗时 {(t_end - t_start)*1000:.2f}ms")

    res = InstanceSegResponse()
    res.seg = ros_numpy.msgify(Image, label_mask, encoding="mono8")
    res.classes = class_map
    return res


def pseudo_seg(req):
    img: np.ndarray = ros_numpy.numpify(req.input)
    # np.save("../tests/gazebo4.npy",img)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    color_num = 5

    # H: 0-179, S: 0-255, V: 0-255
    color_ranges = [
        [np.array([170, 60, 60]), np.array([179, 255, 255])],  # red
        [np.array([90, 60, 60]), np.array([105, 255, 255])],  # blue
        [np.array([10, 60, 60]), np.array([33, 255, 255])],  # yellow
        [np.array([33, 40, 40]), np.array([55, 255, 255])],  # green
        [np.array([105, 60, 60]), np.array([170, 255, 255])],  # purple
    ]

    color_masks = []
    for i in range(color_num):
        mask = cv2.inRange(hsv_img, color_ranges[i][0], color_ranges[i][1])
        if i == 0:
            mask += cv2.inRange(
                hsv_img, np.array([0, 60, 60]), np.array([10, 255, 255])
            )
        color_masks.append(mask)

    output = np.zeros(img.shape[:-1], dtype=np.uint8)  # 用于保存分割结果
    obj_class = [0]
    obj_id = 1
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    for i in range(color_num):
        mask = color_masks[i]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=2)
        mask = cv2.dilate(mask, kernel2, iterations=1)
        num_objs, labels = cv2.connectedComponents(mask)
        for j in range(num_objs - 1):
            output[np.where(np.logical_and((labels == j + 1), (output == 0)))] = obj_id
            print(
                f"第 {obj_id} 个实例属于第 {i+1} 种类别,共 {len(np.where(labels == j + 1)[0])} 个点"
            )
            obj_id += 1
            obj_class.append(i + 1)
            # breakpoint()
    # np.save("../tests/seg.npy",output)

    print(f"实例分割完成,检测到 {obj_id-1} 个实例，属于 {len(set(obj_class))-1} 种类别")
    res = InstanceSegResponse()
    res.seg = ros_numpy.msgify(Image, output, encoding="mono8")
    res.classes = obj_class
    return res


if __name__ == "__main__":
    pseudo = True

    if not pseudo:
        model_path = "../models/model_25_coco.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.logdebug(f"加载实例分割网络 {model_path}")
        seg = Segmentation(model_path, device)

    rospy.init_node("maskrcnn_ros_server")
    if pseudo:
        s = rospy.Service("/maskrcnn_ros_server/seg_instance", InstanceSeg, pseudo_seg)
    else:
        s = rospy.Service(
            "/maskrcnn_ros_server/seg_instance", InstanceSeg, seg_callback
        )
    rospy.loginfo("/maskrcnn_ros_server 节点就绪")
    rospy.spin()
