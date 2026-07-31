import numpy as np
from scipy.spatial.transform import Rotation

class RandomRigidTransform:
    def __init__(self, mag_trans=0.5, mag_rot=45):
        self.mag_trans = mag_trans  # Translation magnitude in meters
        self.mag_rot = mag_rot      # Rotation magnitude in degrees

    def __call__(self, points):
        # Random rotation
        angle_x = np.random.uniform(-self.mag_rot, self.mag_rot)
        angle_y = np.random.uniform(-self.mag_rot, self.mag_rot)
        angle_z = np.random.uniform(-self.mag_rot, self.mag_rot)
        rotation = Rotation.from_euler('xyz', [angle_x, angle_y, angle_z], degrees=True)
        R = rotation.as_matrix()

        # Random translation
        t = np.random.uniform(-self.mag_trans, self.mag_trans, size=3)

        # Build 4x4 transformation matrix
        transform = np.identity(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        # Apply transformation
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points = (transform @ points_homogeneous.T).T[:, :3]

        return transformed_points, transform


class PointCloudAugmentor:
    def __init__(self, mag_trans=0.1, mag_rot=30, jitter_std=0.01, dropout_rate=0.1):
        self.rigid_transform = RandomRigidTransform(mag_trans=mag_trans, mag_rot=mag_rot)
        self.jitter_std = jitter_std
        self.dropout_rate = dropout_rate

    def _jitter(self, points):
        noise = np.random.normal(0.0, self.jitter_std, size=points.shape)
        return points + noise

    def _dropout(self, points):
        if points.shape[0] <= 8 or self.dropout_rate <= 0:
            return points

        keep_mask = np.random.rand(points.shape[0]) > self.dropout_rate
        if keep_mask.sum() < 8:
            keep_mask[np.random.choice(points.shape[0], 8, replace=False)] = True
        return points[keep_mask]

    def _resample(self, points, target_num_points):
        if points.shape[0] == target_num_points:
            return points
        replace = points.shape[0] < target_num_points
        indices = np.random.choice(points.shape[0], target_num_points, replace=replace)
        return points[indices]

    def __call__(self, points, target_num_points=None):
        transformed_points, _ = self.rigid_transform(points)
        transformed_points = self._jitter(transformed_points)
        transformed_points = self._dropout(transformed_points)
        if target_num_points is not None:
            transformed_points = self._resample(transformed_points, target_num_points)
        return transformed_points
