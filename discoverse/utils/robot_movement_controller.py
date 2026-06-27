"""
Pose controller for the MMK2 differential-drive base.

The controller computes a body twist first, then maps it to the two wheel
commands used by the MMK2 MuJoCo actuators.  Set command_mode="twist" if a
different environment expects action[:2] to be [linear_velocity, yaw_rate].
"""

import numpy as np


class RobotMovementController:
    """Stable go-to-point / go-to-pose controller for a differential base."""

    def __init__(
        self,
        max_speed=0.6,
        stationary_thresh=0.05,
        turn_speed=3.5,
        wheel_radius=0.0838,
        wheel_distance=0.189,
        max_wheel_speed=12.0,
        max_wheel_torque=25.0,
        wheel_kp=3.0,
        stop_kp=0.8,
        max_stop_torque=6.0,
        command_mode="wheel",
        kp_dist=1.2,
        kp_heading=2.4,
        kp_final_yaw=2.0,
        max_accel=0.8,
        max_yaw_accel=5.0,
        yaw_thresh=0.05,
        turn_in_place_thresh=0.55,
        settle_steps=3,
    ):
        self.max_speed = max_speed
        self.stationary_thresh = stationary_thresh
        self.turn_speed = turn_speed
        self.wheel_radius = wheel_radius
        self.wheel_distance = wheel_distance
        self.max_wheel_speed = max_wheel_speed
        self.max_wheel_torque = max_wheel_torque
        self.wheel_kp = wheel_kp
        self.stop_kp = stop_kp
        self.max_stop_torque = max_stop_torque
        self.command_mode = command_mode
        self.kp_dist = kp_dist
        self.kp_heading = kp_heading
        self.kp_final_yaw = kp_final_yaw
        self.max_accel = max_accel
        self.max_yaw_accel = max_yaw_accel
        self.yaw_thresh = yaw_thresh
        self.turn_in_place_thresh = turn_in_place_thresh
        self.settle_steps = settle_steps

        self._last_v = 0.0
        self._last_w = 0.0
        self._last_goal_key = None
        self._settle_count = 0

    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def reset(self):
        self._last_v = 0.0
        self._last_w = 0.0
        self._last_goal_key = None
        self._settle_count = 0

    def _get_robot_pose(self, sim_node, obs=None):
        """Get robot xy position and yaw angle in the world frame."""
        robot_pos = np.asarray(sim_node.mj_data.qpos[:2], dtype=float)
        quat = np.asarray(sim_node.mj_data.qpos[3:7], dtype=float)  # [w, x, y, z]

        if not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
            quat = np.asarray(obs.get("base_orientation", [1.0, 0.0, 0.0, 0.0]), dtype=float)

        quat = quat / (np.linalg.norm(quat) + 1e-12)
        yaw = np.arctan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
        return robot_pos, yaw

    def _goal_key(self, target_pos, target_yaw, allow_reverse):
        yaw_key = None if target_yaw is None else round(float(target_yaw), 3)
        return (
            round(float(target_pos[0]), 3),
            round(float(target_pos[1]), 3),
            yaw_key,
            bool(allow_reverse),
        )

    def _mark_goal_active(self, goal_key):
        if goal_key != self._last_goal_key:
            self._last_goal_key = goal_key
            self._settle_count = 0

    def _limit_twist(self, sim_node, v, w):
        dt = float(getattr(sim_node, "delta_t", 0.02))
        dt = max(dt, 1e-3)

        v = float(np.clip(v, -self.max_speed, self.max_speed))
        w = float(np.clip(w, -self.turn_speed, self.turn_speed))

        max_dv = self.max_accel * dt
        max_dw = self.max_yaw_accel * dt
        v = self._last_v + np.clip(v - self._last_v, -max_dv, max_dv)
        w = self._last_w + np.clip(w - self._last_w, -max_dw, max_dw)

        self._last_v = v
        self._last_w = w
        return v, w

    def _set_base_command(self, sim_node, action, v, w):
        v, w = self._limit_twist(sim_node, v, w)

        if self.command_mode == "twist":
            action[:2] = [v, w]
            return action

        wheel_radius = float(getattr(sim_node, "wheel_radius", self.wheel_radius))
        wheel_distance = float(getattr(sim_node, "wheel_distance", self.wheel_distance))

        desired_wheel_vel = np.array([
            (v - 0.5 * wheel_distance * w) / wheel_radius,
            (v + 0.5 * wheel_distance * w) / wheel_radius,
        ])
        desired_wheel_vel = np.clip(desired_wheel_vel, -self.max_wheel_speed, self.max_wheel_speed)

        # MMK2 wheel actuators are torque motors, not velocity actuators.
        # Close a small velocity loop here so twist commands behave predictably.
        current_wheel_vel = np.asarray(getattr(sim_node, "sensor_wheel_qvel", [0.0, 0.0])[:2], dtype=float)
        wheel_torque = self.wheel_kp * (desired_wheel_vel - current_wheel_vel)
        action[:2] = np.clip(wheel_torque, -self.max_wheel_torque, self.max_wheel_torque)
        return action

    def stop(self, sim_node, action):
        self._last_v = 0.0
        self._last_w = 0.0
        if self.command_mode == "wheel":
            current_wheel_vel = np.asarray(getattr(sim_node, "sensor_wheel_qvel", [0.0, 0.0])[:2], dtype=float)
            action[:2] = np.clip(-self.stop_kp * current_wheel_vel, -self.max_stop_torque, self.max_stop_torque)
        else:
            action[:2] = [0.0, 0.0]
        return action

    def hold_pose(
        self,
        sim_node,
        obs,
        action,
        target_pos,
        target_yaw,
        position_thresh=0.015,
        yaw_thresh=0.035,
        max_speed=0.12,
        max_yaw_rate=0.4,
    ):
        """Keep the mobile base at a fixed pose while the upper body is moving."""
        target_pos = np.asarray(target_pos[:2], dtype=float)
        robot_pos, yaw = self._get_robot_pose(sim_node, obs)

        diff = target_pos - robot_pos
        yaw_err = self._wrap_angle(float(target_yaw) - yaw)
        if np.linalg.norm(diff) < position_thresh and abs(yaw_err) < yaw_thresh:
            return self.stop(sim_node, action)

        heading = np.array([np.cos(yaw), np.sin(yaw)])
        lateral = np.array([-np.sin(yaw), np.cos(yaw)])
        forward_err = float(np.dot(diff, heading))
        lateral_err = float(np.dot(diff, lateral))

        v = np.clip(2.0 * forward_err, -max_speed, max_speed)
        w = np.clip(2.0 * yaw_err + 1.5 * lateral_err, -max_yaw_rate, max_yaw_rate)
        return self._set_base_command(sim_node, action, v, w)

    def drive_velocity(self, sim_node, action, linear_velocity, yaw_rate):
        """Command a base twist directly."""
        return self._set_base_command(sim_node, action, linear_velocity, yaw_rate)

    def move_to_pose(
        self,
        sim_node,
        obs,
        action,
        target_pos,
        target_yaw=None,
        position_thresh=None,
        yaw_thresh=None,
        allow_reverse=False,
    ):
        """Move to target (x, y), then optionally align to target_yaw."""
        target_pos = np.asarray(target_pos[:2], dtype=float)
        position_thresh = self.stationary_thresh if position_thresh is None else position_thresh
        yaw_thresh = self.yaw_thresh if yaw_thresh is None else yaw_thresh

        goal_key = self._goal_key(target_pos, target_yaw, allow_reverse)
        self._mark_goal_active(goal_key)

        robot_pos, yaw = self._get_robot_pose(sim_node, obs)
        diff = target_pos - robot_pos
        dist = float(np.linalg.norm(diff))

        pos_done = dist < position_thresh
        yaw_done = True
        yaw_err = 0.0
        if target_yaw is not None:
            yaw_err = self._wrap_angle(float(target_yaw) - yaw)
            yaw_done = abs(yaw_err) < yaw_thresh

        if pos_done and yaw_done:
            self._settle_count += 1
            action = self.stop(sim_node, action)
            return action, self._settle_count >= self.settle_steps

        self._settle_count = 0

        if not pos_done:
            path_yaw = np.arctan2(diff[1], diff[0])
            forward_err = self._wrap_angle(path_yaw - yaw)
            reverse_err = self._wrap_angle(path_yaw + np.pi - yaw)
            use_reverse = allow_reverse and abs(reverse_err) < abs(forward_err)

            heading_err = reverse_err if use_reverse else forward_err
            direction = -1.0 if use_reverse else 1.0

            if abs(heading_err) > self.turn_in_place_thresh:
                v = 0.0
            else:
                heading_scale = max(0.15, np.cos(heading_err))
                v = direction * min(self.kp_dist * dist, self.max_speed) * heading_scale

            w = self.kp_heading * heading_err
            action = self._set_base_command(sim_node, action, v, w)
            return action, False

        w = self.kp_final_yaw * yaw_err
        action = self._set_base_command(sim_node, action, 0.0, w)
        return action, False

    def move_to_face(
        self,
        sim_node,
        obs,
        action,
        target_pos,
        face_pos,
        position_thresh=None,
        yaw_thresh=None,
        allow_reverse=False,
    ):
        """Move to target_pos and finish facing face_pos."""
        target_pos = np.asarray(target_pos[:2], dtype=float)
        face_pos = np.asarray(face_pos[:2], dtype=float)
        face_vec = face_pos - target_pos

        target_yaw = None
        if np.linalg.norm(face_vec) > 1e-6:
            target_yaw = np.arctan2(face_vec[1], face_vec[0])

        return self.move_to_pose(
            sim_node,
            obs,
            action,
            target_pos,
            target_yaw=target_yaw,
            position_thresh=position_thresh,
            yaw_thresh=yaw_thresh,
            allow_reverse=allow_reverse,
        )

    def move_to_xy(
        self,
        sim_node,
        obs,
        action,
        target_pos,
        final_yaw=None,
        allow_reverse=False,
        position_thresh=None,
        yaw_thresh=None,
    ):
        """Move robot base toward target (x, y) position."""
        return self.move_to_pose(
            sim_node,
            obs,
            action,
            target_pos,
            target_yaw=final_yaw,
            allow_reverse=allow_reverse,
            position_thresh=position_thresh,
            yaw_thresh=yaw_thresh,
        )

    def turn_to_yaw(self, sim_node, obs, action, target_yaw, yaw_thresh=None, min_yaw_rate=0.0):
        """Turn in place to a world-frame yaw angle."""
        _, yaw = self._get_robot_pose(sim_node, obs)
        yaw_thresh = self.yaw_thresh if yaw_thresh is None else yaw_thresh
        yaw_err = self._wrap_angle(float(target_yaw) - yaw)

        goal_key = ("turn", round(float(target_yaw), 3))
        self._mark_goal_active(goal_key)

        if abs(yaw_err) < yaw_thresh:
            self._settle_count += 1
            action = self.stop(sim_node, action)
            return action, self._settle_count >= self.settle_steps

        self._settle_count = 0
        w = self.kp_final_yaw * yaw_err
        if min_yaw_rate > 0.0 and abs(w) < min_yaw_rate:
            w = np.sign(yaw_err) * min_yaw_rate
        action = self._set_base_command(sim_node, action, 0.0, w)
        return action, False

    def turn_to_face(self, sim_node, obs, action, target_pos, yaw_thresh=None, min_yaw_rate=0.0):
        """Turn robot to face target position."""
        robot_pos, _ = self._get_robot_pose(sim_node, obs)
        target_pos = np.asarray(target_pos[:2], dtype=float)
        diff = target_pos - robot_pos

        if np.linalg.norm(diff) < 1e-6:
            action = self.stop(sim_node, action)
            return action, True

        target_yaw = np.arctan2(diff[1], diff[0])
        return self.turn_to_yaw(
            sim_node,
            obs,
            action,
            target_yaw,
            yaw_thresh=yaw_thresh,
            min_yaw_rate=min_yaw_rate,
        )
