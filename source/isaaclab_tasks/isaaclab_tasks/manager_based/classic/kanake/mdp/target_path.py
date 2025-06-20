# target_path.py

import numpy as np


WAYPOINTS = [
    (0.3605, -0.09937),
    (0.68786, -0.09937),
    (0.92616, -0.09937),
    (1.20626, -0.09937),
    (1.21200, -0.36058),
    (1.21257, -0.59212),
]

class TargetPathManager:
    def __init__(self, waypoints=None, threshold=0.05):
        if waypoints is None:
            waypoints = WAYPOINTS
        self.waypoints = np.array(waypoints)
        self.threshold = threshold
        self.current_target_idx = 0

    def reset(self):
        self.current_target_idx = 0

    def update(self, robot_pos):
        target_point = self.waypoints[self.current_target_idx]
        distance = np.linalg.norm(robot_pos - target_point)

        if distance < self.threshold and self.current_target_idx < len(self.waypoints) - 1:
            self.current_target_idx += 1

        return self.current_target_idx

    def get_target_point(self):
        return self.waypoints[self.current_target_idx]
