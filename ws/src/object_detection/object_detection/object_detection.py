from abc import ABC, abstractmethod
from threading import Lock
from typing import List, Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO, YOLOE

from pepper_interfaces.msg import Detection, Detections
from pepper_interfaces.srv import ObjectDetection


# --------------------------------------------------------------------------- #
#  Detector back-ends
# --------------------------------------------------------------------------- #
class BaseDetector(ABC):
    """Common interface: every detector returns an Ultralytics Results object."""

    def __init__(self, model_path: str, conf: float, logger):
        self.conf = conf
        self.logger = logger
        self.model_path = model_path

    @abstractmethod
    def predict(self, img, classes: Optional[List[str]] = None):
        """Run detection on `img`. `classes` meaning depends on the detector."""


class ClassicDetector(BaseDetector):
    """Closed-set detector (YOLO): the label set is fixed at training time."""

    def __init__(self, model_path: str, conf: float, logger):
        super().__init__(model_path, conf, logger)
        self.model = YOLO(model_path)
        self.name2id = {name: idx for idx, name in self.model.names.items()}

    def predict(self, img, classes: Optional[List[str]] = None):
        class_ids = None
        if classes:
            # Classes act as a filter on the fixed label set
            unknown = [c for c in classes if c not in self.name2id]
            if unknown:
                self.logger.warn(
                    f"Classes {unknown} are not known by the classic detector "
                    f"({self.model_path}) and will be ignored. "
                    "Use the open-vocabulary detector to find them.")
            class_ids = [self.name2id[c] for c in classes if c in self.name2id]
            if not class_ids:
                class_ids = None  # nothing valid -> return everything
        return self.model.predict(img, conf=self.conf, classes=class_ids, verbose=False)[0]


class OpenVocabDetector(BaseDetector):
    """Open-vocabulary detector (YOLOE): the label set is given as text prompts."""

    def __init__(self, model_path: str, conf: float, logger, default_classes: List[str]):
        super().__init__(model_path, conf, logger)
        self.model = YOLOE(model_path)
        self.default_classes = list(default_classes)
        self.current_classes: Optional[List[str]] = None
        self._set_classes(self.default_classes)

    def _set_classes(self, classes: List[str]):
        # Encoding the prompts with the text encoder is expensive:
        # do it only when the requested vocabulary changes.
        if classes == self.current_classes:
            return
        self.logger.info(f"Setting open-vocabulary classes: {classes}")
        self.model.set_classes(classes, self.model.get_text_pe(classes))
        self.current_classes = list(classes)

    def predict(self, img, classes: Optional[List[str]] = None):
        self._set_classes(list(classes) if classes else self.default_classes)
        return self.model.predict(img, conf=self.conf, verbose=False)[0]


DETECTORS = {
    "classic": ClassicDetector,
    "open_vocab": OpenVocabDetector,
}


# --------------------------------------------------------------------------- #
#  ROS2 node
# --------------------------------------------------------------------------- #
class ObjDetector(Node):
    NODE_NAME = "object_detector_node"

    def __init__(self):
        super().__init__(self.NODE_NAME)

        # ---- Parameters (see config/object_detector_conf.yaml) ----
        self.declare_parameter("detector_type", "classic")
        self.declare_parameter("classic_model", "yolo11n.pt")
        self.declare_parameter("open_vocab_model", "yoloe-11s-seg.pt")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("classes", ["person"])

        detector_type = self.get_parameter("detector_type").value
        conf = float(self.get_parameter("conf").value)
        classes = list(self.get_parameter("classes").value)

        if detector_type not in DETECTORS:
            raise ValueError(
                f"Unknown detector_type '{detector_type}'. Valid values: {list(DETECTORS)}")

        if detector_type == "classic":
            model_path = self.get_parameter("classic_model").value
            self.detector = ClassicDetector(model_path, conf, self.get_logger())
        else:
            model_path = self.get_parameter("open_vocab_model").value
            self.detector = OpenVocabDetector(model_path, conf, self.get_logger(), classes)
        self.detector_type = detector_type

        # The topic callback and the service callback belong to different
        # callback groups, so they may run in parallel: the lock makes
        # "set classes + predict" atomic.
        self.lock = Lock()
        self.add_on_set_parameters_callback(self._on_param_change)

        # ---- ROS interfaces ----
        self.br = CvBridge()
        self.pub_detections = self.create_publisher(Detections, "/in_rgb/detect", qos_profile=10)
        self.pub_image = self.create_publisher(Image, "/in_rgb/view", qos_profile=10)
        self.detect_srv = self.create_service(
            ObjectDetection, "detect_objects", self.detect_callback,
            callback_group=MutuallyExclusiveCallbackGroup())
        self.sub_image = self.create_subscription(
            Image, "/in_rgb", self.detect, qos_profile=1,
            callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(
            f"Object Detector Node started: type='{detector_type}', model='{model_path}'")

    def _on_param_change(self, params):
        """Allow changing the default open-vocabulary prompts at runtime."""
        for p in params:
            if p.name == "classes" and isinstance(self.detector, OpenVocabDetector):
                if not p.value:
                    return SetParametersResult(successful=False, reason="classes cannot be empty")
                with self.lock:
                    self.detector.default_classes = list(p.value)
                self.get_logger().info(f"Default classes updated: {list(p.value)}")
            elif p.name in ("detector_type", "classic_model", "open_vocab_model", "conf"):
                return SetParametersResult(
                    successful=False, reason=f"'{p.name}' can only be set at startup")
        return SetParametersResult(successful=True)

    def detect(self, img_msg: Image, classes: Optional[List[str]] = None) -> List[Detection]:
        img = self.br.imgmsg_to_cv2(img_msg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.lock:
            predict = self.detector.predict(img, classes)
        detections = self._predict2detect_msg(predict)
        self.pub_detections.publish(Detections(detections=detections))
        nw_img = self._draw_boxes(img, predict)
        self.pub_image.publish(self.br.cv2_to_imgmsg(nw_img, encoding="rgb8"))
        return detections

    def detect_callback(self, request: ObjectDetection.Request,
                        response: ObjectDetection.Response):
        classes = list(request.classes)
        self.get_logger().info(
            f"Detection request ({self.detector_type}), classes={classes or 'default'}")
        response.detections = self.detect(request.image, classes)
        return response

    @staticmethod
    def _predict2detect_msg(predict) -> List[Detection]:
        boxes = predict.boxes.xyxy.tolist()
        class_ids = list(map(int, predict.boxes.cls.tolist()))
        scores = predict.boxes.conf.tolist()
        names = predict.names
        assert len(boxes) == len(class_ids), "Mismatch between boxes and class IDs"

        detections = []
        for box, class_id, score in zip(boxes, class_ids, scores):
            det = Detection()
            det.class_id = class_id
            det.class_name = names[class_id]
            det.box = [float(v) for v in box]
            det.score = float(score)
            detections.append(det)
        return detections

    @staticmethod
    def _draw_boxes(org_img, predict):
        nw_img = org_img.copy()
        names = predict.names
        for box, class_id, score in zip(predict.boxes.xyxy.tolist(),
                                        predict.boxes.cls.tolist(),
                                        predict.boxes.conf.tolist()):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(class_id)]} {score:.2f}"
            cv2.rectangle(nw_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.putText(nw_img, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 0, 0), thickness=2)
        return nw_img


def test():
    """Call the service on bus.jpg, first with the default classes, then with custom ones."""
    import sys
    from pathlib import Path
    from threading import Thread

    curr_dir = Path(__file__).parent
    rclpy.init()
    node = ObjDetector()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    Thread(target=executor.spin, daemon=True).start()

    image = cv2.imread(curr_dir.joinpath("bus.jpg").as_posix())
    assert image is not None, "Image not found"
    client = node.create_client(ObjectDetection, "detect_objects")
    client.wait_for_service()

    for classes in ([], ["person", "sneakers", "glasses"]):
        req = ObjectDetection.Request()
        req.image = node.br.cv2_to_imgmsg(image)
        req.classes = classes
        resp = client.call(req)
        print(f"\nclasses={classes or 'default'}")
        for d in resp.detections:
            print(f"  {d.class_name:<12} {d.score:.2f} {[round(v) for v in d.box]}")
    sys.exit()


def main():
    rclpy.init()
    node = ObjDetector()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    test()
