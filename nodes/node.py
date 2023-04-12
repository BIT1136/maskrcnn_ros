#!/usr/bin/env python

import time

import torch
import numpy as np

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
    _, predict_classes, _, predict_masks = seg.predict(img)
    label_mask = np.zeros(shape[:2], dtype=np.uint8)
    mask_threshold = 0.5
    num_instances = len(predict_classes)
    for i in range(num_instances):
        index = np.where(predict_masks[i] > mask_threshold)
        label_mask[index] = i + 1

    t_end = time.perf_counter()
    rospy.loginfo(f"实例分割完成,检测到 {num_instances} 个实例,耗时 {(t_end - t_start)*1000:.2f}ms")

    res = InstanceSegResponse()
    res.seg = ros_numpy.msgify(Image, label_mask, encoding="mono8")
    return res


if __name__ == "__main__":
    model_path = "../models/model_19.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rospy.logdebug(f"加载实例分割网络 {model_path}")
    seg = Segmentation(model_path, device)

    rospy.init_node("maskrcnn_ros_server")
    s = rospy.Service("/maskrcnn_ros_server/seg_instance", InstanceSeg, seg_callback)
    rospy.loginfo("/maskrcnn_ros_server 节点就绪")
    rospy.spin()
