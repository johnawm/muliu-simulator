import logging
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.utils import SimpleStateMachine, RobotMovementController, step_func

logger = logging.getLogger(__name__)


class BoxTransport:
    """Task-level state machine for transporting one box with the MMK2 robot."""

    STATE_INIT = 0
    STATE_ALIGN_Y = 1
    STATE_APPROACH = 2
    STATE_EXTEND_ARMS = 3
    STATE_GRASP = 4
    STATE_LIFT = 5
    STATE_BACKUP = 6
    STATE_TURN_AROUND = 7
    STATE_ADJUST_HEIGHT = 8
    STATE_MOVE_TO_TARGET = 9
    STATE_DESCEND = 10
    STATE_RELEASE = 11
    BASE_MOVING_STATES = {
        STATE_ALIGN_Y,
        STATE_APPROACH,
        STATE_BACKUP,
        STATE_TURN_AROUND,
        STATE_MOVE_TO_TARGET,
        STATE_RELEASE,
    }

    def __init__(
        self,
        box: dict,
        box_target_pos: np.ndarray,
        box_size: np.ndarray,
        approach_distance: float = 0.5,
        lift_height: float = 0.1,
        release_height_offset: float = 0.1,
        move_speed: float = 1.0,
        move_init_speed: float = 5.0,
    ):
        self.box_name = box["name"]
        self.box_original_pos = np.array(box["original_pos"], dtype=float)
        self.box_target_pos = np.array(box_target_pos, dtype=float)
        self.box_size = np.array(box_size, dtype=float)

        self.approach_distance = float(approach_distance)
        self.lift_height = float(lift_height)
        self.release_height_offset = float(release_height_offset)
        self.move_speed = float(move_speed)
        self.move_init_speed = float(move_init_speed)

        self.move_controller = RobotMovementController()

        self.state_machine = SimpleStateMachine()
        self.state_machine.max_state_cnt = 12
        self.state_data = {}

        self.lft_arm_rot = Rotation.from_euler("zyx", [np.pi / 2, -0.0551 + np.pi, np.pi / 8]).as_matrix()
        self.rgt_arm_rot = Rotation.from_euler("zyx", [-np.pi / 2, -0.0551 + np.pi, -np.pi / 8]).as_matrix()

        logger.info(
            "BoxTransport initialized: name=%s, original=%s, target=%s, size=%s",
            self.box_name,
            self.box_original_pos,
            self.box_target_pos,
            self.box_size,
        )

    def reset(self):
        self.state_machine.reset()
        self.state_data = {}
        self.move_controller.reset()
        logger.info("BoxTransport reset")

    def is_complete(self) -> bool:
        return self.state_machine.state_idx >= self.state_machine.max_state_cnt

    @staticmethod
    def _yaw_from_to(src_xy, dst_xy) -> float:
        diff = np.asarray(dst_xy[:2], dtype=float) - np.asarray(src_xy[:2], dtype=float)
        return float(np.arctan2(diff[1], diff[0]))

    def _pickup_distance(self) -> float:
        # Keep the base in front of the pallet. In pick_box.xml the pallet
        # starts near x=2.45, so driving closer than this causes chassis contact.
        return min(max(self.approach_distance, 0.78), 0.82)

    def _pickup_stage_xy(self) -> np.ndarray:
        stage_distance = max(self._pickup_distance() + 0.08, 0.90)
        return self.box_original_pos[:2] + np.array([-stage_distance, 0.0])

    def _pickup_xy(self) -> np.ndarray:
        return self.box_original_pos[:2] + np.array([-self._pickup_distance(), 0.0])

    def _current_box_pos(self, sim_node) -> np.ndarray:
        qid = getattr(sim_node, "free_body_qpos_ids", {}).get(self.box_name)
        if qid is None:
            logger.warning("No qpos id for box %s", self.box_name)
            return self.box_original_pos.copy()
        qpos_adr = sim_node.mj_model.jnt_qposadr[qid]
        return sim_node.mj_data.qpos[qpos_adr:qpos_adr + 3].copy()

    def _box_side_offset(self, clearance: float) -> float:
        return 0.5 * float(self.box_size[2]) + clearance

    def _box_top_offset(self, clearance: float) -> float:
        return 0.5 * float(self.box_size[0]) + clearance

    def _release_hand_z(self) -> float:
        return float(self.box_target_pos[2] + 0.5 * self.box_size[2])

    def _slide_ctrl_range(self, sim_node) -> Tuple[float, float]:
        try:
            slide_id = sim_node.mj_model.joint("slide_joint").id
            lo, hi = sim_node.mj_model.jnt_range[slide_id]
            return float(lo), float(hi)
        except Exception:
            return -0.04, 0.87

    def _box_front_offset(self, clearance: float) -> float:
        return 0.5 * float(self.box_size[1]) + clearance

    def _pickup_yaw(self) -> float:
        return self._yaw_from_to(self._pickup_xy(), self.box_original_pos[:2])

    def _pickup_alignment_error(self, sim_node):
        robot_pos, yaw = self.move_controller._get_robot_pose(sim_node)
        pickup_xy = self._pickup_xy()
        return {
            "x": float(robot_pos[0] - pickup_xy[0]),
            "y": float(robot_pos[1] - pickup_xy[1]),
            "yaw": float(self.move_controller._wrap_angle(yaw - self._pickup_yaw())),
        }

    def _pickup_alignment_ready(self, sim_node) -> bool:
        err = self._pickup_alignment_error(sim_node)
        ready = abs(err["x"]) < 0.04 and abs(err["y"]) < 0.025 and abs(err["yaw"]) < 0.04
        logger.info("ALIGNMENT: x=%.3f/0.04 y=%.3f/0.025 yaw=%.3f/0.04 -> ready=%s",
                    err["x"], err["y"], err["yaw"], ready)
        return ready

    def _release_xy(self) -> np.ndarray:
        # The shelf is on the negative-x side; keep the base in front of it.
        # Robot stops at box_target_pos + approach_distance on x-axis, then extends arm to place box.
        return self.box_target_pos[:2] + np.array([self.approach_distance, 0.0])

    def _backup_clear_xy(self, sim_node) -> np.ndarray:
        robot_pos, _ = self.move_controller._get_robot_pose(sim_node)
        clear_distance = max(self._pickup_distance() + 0.35, 1.15)
        clear_xy = self.box_original_pos[:2] + np.array([-clear_distance, 0.0])
        clear_x = min(clear_xy[0], robot_pos[0] - 0.2)
        return np.array([clear_x, clear_xy[1]], dtype=float)

    def _align_lane_yaw(self, sim_node) -> float:
        robot_pos, _ = self.move_controller._get_robot_pose(sim_node)
        y_err = float(self.box_original_pos[1] - robot_pos[1])
        if abs(y_err) < 0.05:
            return self._pickup_yaw()
        return self._pickup_yaw() + np.sign(y_err) * 1.2

    def _record_base_hold_pose(self, sim_node):
        robot_pos, yaw = self.move_controller._get_robot_pose(sim_node)
        self.state_data["hold_base_xy"] = robot_pos.copy()
        self.state_data["hold_base_yaw"] = yaw

    def _hold_base_pose(self, sim_node, obs, action):
        if "hold_base_xy" not in self.state_data or "hold_base_yaw" not in self.state_data:
            self._record_base_hold_pose(sim_node)
        return self.move_controller.hold_pose(
            sim_node,
            obs,
            action,
            self.state_data["hold_base_xy"],
            self.state_data["hold_base_yaw"],
        )

    def _move_to_face(self, sim_node, obs, action, target_xy, face_xy, position_thresh=0.04, yaw_thresh=0.04):
        return self.move_controller.move_to_face(
            sim_node,
            obs,
            action,
            target_xy,
            face_xy,
            position_thresh=position_thresh,
            yaw_thresh=yaw_thresh,
        )

    def _state_uses_base_motion(self, state) -> bool:
        if state in self.BASE_MOVING_STATES:
            return True
        if state == self.STATE_EXTEND_ARMS and not self.state_data.get("extend_ready", True):
            return True
        return False

    def _enter_state(self, sim_node, action):
        state = self.state_machine.state_idx
        logger.info("Entering state %d (%s)", state, self.get_state_name(state))

        self.move_controller.stop(sim_node, action)
        if not self._state_uses_base_motion(state):
            self._record_base_hold_pose(sim_node)

        if state == self.STATE_INIT:
            self._init_state_enter(sim_node)
        elif state == self.STATE_ALIGN_Y:
            self._align_y_state_enter(sim_node)
        elif state == self.STATE_APPROACH:
            self._approach_state_enter(sim_node)
        elif state == self.STATE_EXTEND_ARMS:
            self._extend_arms_state_enter(sim_node)
        elif state == self.STATE_GRASP:
            self._grasp_state_enter(sim_node)
        elif state == self.STATE_LIFT:
            self._lift_state_enter(sim_node)
        elif state == self.STATE_BACKUP:
            self._backup_state_enter(sim_node)
        elif state == self.STATE_ADJUST_HEIGHT:
            self._adjust_height_state_enter(sim_node)
        elif state == self.STATE_TURN_AROUND:
            self._turn_around_state_enter(sim_node)
        elif state == self.STATE_MOVE_TO_TARGET:
            self._move_to_target_state_enter(sim_node)
        elif state == self.STATE_DESCEND:
            self._descend_state_enter(sim_node)
        elif state == self.STATE_RELEASE:
            self._release_state_enter(sim_node)

    def _execute_state_logic(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        state = self.state_machine.state_idx

        if state == self.STATE_INIT:
            return self._init_state_execute(sim_node, obs, action)
        if state == self.STATE_ALIGN_Y:
            return self._align_y_state_execute(sim_node, obs, action)
        if state == self.STATE_APPROACH:
            return self._approach_state_execute(sim_node, obs, action)
        if state == self.STATE_EXTEND_ARMS:
            return self._extend_arms_state_execute(sim_node, obs, action)
        if state == self.STATE_GRASP:
            return self._grasp_state_execute(sim_node, obs, action)
        if state == self.STATE_LIFT:
            return self._lift_state_execute(sim_node, obs, action)
        if state == self.STATE_BACKUP:
            return self._backup_state_execute(sim_node, obs, action)
        if state == self.STATE_ADJUST_HEIGHT:
            return self._adjust_height_state_execute(sim_node, obs, action)
        if state == self.STATE_TURN_AROUND:
            return self._turn_around_state_execute(sim_node, obs, action)
        if state == self.STATE_MOVE_TO_TARGET:
            return self._move_to_target_state_execute(sim_node, obs, action)
        if state == self.STATE_DESCEND:
            return self._descend_state_execute(sim_node, obs, action)
        if state == self.STATE_RELEASE:
            return self._release_state_execute(sim_node, obs, action)

        return action, True

    def _init_state_enter(self, sim_node):
        sim_node.tctr_head[1] = -0.4
        current_box_pos = self._current_box_pos(sim_node)
        logger.info("INIT: current_box_pos=%s", current_box_pos)
        slide_height = 1.22 - 1.08 * current_box_pos[2]
        sim_node.tctr_slide[0] = slide_height
        sim_node.tctr_left_arm[:] = sim_node.init_joint_ctrl[5:11]
        sim_node.tctr_right_arm[:] = sim_node.init_joint_ctrl[12:18]
        sim_node.tctr_lft_gripper[:] = 0
        sim_node.tctr_rgt_gripper[:] = 0
        sim_node.lft_arm_target_pose[:] = sim_node.arm_action_init_position[0]
        sim_node.rgt_arm_target_pose[:] = sim_node.arm_action_init_position[1]
        sim_node.set_left_arm_new_target = False
        sim_node.set_right_arm_new_target = False
        logger.info("INIT: head=-0.4, slide_height=%.4f, retract arms", slide_height)

    def _init_state_execute(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        slide_done = (
            np.allclose(sim_node.tctr_slide, sim_node.sensor_slide_qpos, atol=3e-2)
            and np.abs(sim_node.sensor_slide_qvel).sum() < 1e-2
        )
        head_done = (
            np.allclose(sim_node.tctr_head, sim_node.sensor_head_qpos, atol=3e-2)
            and np.abs(sim_node.sensor_head_qvel).sum() < 0.1
        )
        arms_done = (
            np.allclose(sim_node.tctr_left_arm, sim_node.sensor_lft_arm_qpos, atol=5e-2)
            and np.allclose(sim_node.tctr_right_arm, sim_node.sensor_rgt_arm_qpos, atol=5e-2)
            and np.abs(sim_node.sensor_lft_arm_qvel).sum() < 0.1
            and np.abs(sim_node.sensor_rgt_arm_qvel).sum() < 0.1
        )
        done = slide_done and head_done and arms_done
        if done:
            logger.info("INIT state completed")
        return action, done

    def _align_y_state_enter(self, sim_node):
        self.state_data["pickup_stage_xy"] = self._pickup_stage_xy()
        self.state_data["align_phase"] = "lane_turn"
        self.state_data["align_lane_yaw"] = self._align_lane_yaw(sim_node)
        logger.info(
            "ALIGN_Y: phase=lane_turn, pickup_stage_xy=%s, lane_yaw=%.3f, pickup_yaw=%.3f",
            self.state_data["pickup_stage_xy"],
            self.state_data["align_lane_yaw"],
            self._pickup_yaw(),
        )

    def _align_y_state_execute(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        phase = self.state_data.get("align_phase", "lane_turn")
        if phase == "lane_turn":
            y_err = float(self.box_original_pos[1] - self.move_controller._get_robot_pose(sim_node, obs)[0][1])
            if abs(y_err) < 0.05:
                self.state_data["align_phase"] = "yaw"
                self.move_controller.stop(sim_node, action)
                logger.info("ALIGN_Y: lane skipped, y_err=%.3f", y_err)
                return action, False

            action, turn_done = self.move_controller.turn_to_yaw(
                sim_node,
                obs,
                action,
                self.state_data["align_lane_yaw"],
                yaw_thresh=0.10,
                min_yaw_rate=0.60,
            )
            if turn_done:
                self.state_data["align_phase"] = "lane_drive"
                self.move_controller.stop(sim_node, action)
                robot_pos, yaw = self.move_controller._get_robot_pose(sim_node, obs)
                logger.info("ALIGN_Y: lane_turn completed at pos=%s yaw=%.3f", robot_pos, yaw)
            return action, False

        if phase == "lane_drive":
            robot_pos, yaw = self.move_controller._get_robot_pose(sim_node, obs)
            y_err = float(self.box_original_pos[1] - robot_pos[1])
            lane_yaw = self.state_data["align_lane_yaw"]
            yaw_err = self.move_controller._wrap_angle(lane_yaw - yaw)
            lane_done = abs(y_err) < 0.025
            if lane_done:
                self.state_data["align_phase"] = "yaw"
                self.move_controller.stop(sim_node, action)
                logger.info("ALIGN_Y: lane_drive completed at pos=%s yaw=%.3f y_err=%.3f", robot_pos, yaw, y_err)
            else:
                v = np.clip(0.65 * abs(y_err), 0.10, 0.38)
                action = self.move_controller.drive_velocity(sim_node, action, v, 2.5 * yaw_err)
            return action, False

        action, done = self.move_controller.turn_to_yaw(
            sim_node,
            obs,
            action,
            self._pickup_yaw(),
            yaw_thresh=0.04,
            min_yaw_rate=0.60,
        )

        if done:
            logger.info("ALIGN_Y state completed")
        return action, done

    def _approach_state_enter(self, sim_node):
        self.state_data["pickup_xy"] = self._pickup_xy()
        logger.info("APPROACH: pickup_xy=%s", self.state_data["pickup_xy"])

    def _approach_state_execute(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        action, done = self.move_controller.move_to_pose(
            sim_node,
            obs,
            action,
            self.state_data["pickup_xy"],
            target_yaw=self._pickup_yaw(),
            position_thresh=0.025,
            yaw_thresh=0.035,
        )

        aligned = self._pickup_alignment_ready(sim_node)
        if done and not aligned:
            err = self._pickup_alignment_error(sim_node)
            logger.info("APPROACH fine alignment pending: x=%.3f y=%.3f yaw=%.3f", err["x"], err["y"], err["yaw"])
        done = done and aligned

        if done:
            err = self._pickup_alignment_error(sim_node)
            logger.info("APPROACH state completed: x=%.3f y=%.3f yaw=%.3f", err["x"], err["y"], err["yaw"])
        return action, done

    def _extend_arms_state_enter(self, sim_node):
        if not self._pickup_alignment_ready(sim_node):
            self.state_data["extend_ready"] = False
            err = self._pickup_alignment_error(sim_node)
            logger.warning("EXTEND_ARMS delayed: pickup alignment x=%.3f y=%.3f yaw=%.3f", err["x"], err["y"], err["yaw"])
            return

        self._set_extend_arm_targets(sim_node)

    def _set_extend_arm_targets(self, sim_node):
        self.state_data["extend_ready"] = True
        self._record_base_hold_pose(sim_node)
        box_pos = self.box_original_pos
        side_offset = self._box_side_offset(0.1)
        front_offset = -0.2
        top_offset = self._box_top_offset(-0.01)
        target_posi1 = box_pos + np.array([front_offset, side_offset, top_offset])
        target_posi2 = box_pos + np.array([front_offset, -side_offset, top_offset])

        sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi1)
        sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi2)

        logger.info("EXTEND_ARMS: box_center=%s, box_size=%s, target_posi1=%s, target_posi2=%s, lft_arm_target_pose=%s, rgt_arm_target_pose=%s",
            box_pos, self.box_size, target_posi1, target_posi2, sim_node.lft_arm_target_pose, sim_node.rgt_arm_target_pose)

        sim_node.setArmEndTarget(
            sim_node.lft_arm_target_pose,
            sim_node.arm_action,
            "l",
            sim_node.sensor_lft_arm_qpos,
            self.lft_arm_rot,
        )
        sim_node.setArmEndTarget(
            sim_node.rgt_arm_target_pose,
            sim_node.arm_action,
            "r",
            sim_node.sensor_rgt_arm_qpos,
            self.rgt_arm_rot,
        )

        sim_node.tctr_lft_gripper[:] = 0
        sim_node.tctr_rgt_gripper[:] = 0

    def _extend_arms_state_execute(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        if not self.state_data.get("extend_ready", True):
            pickup_xy = self._pickup_xy()
            logger.info("EXTEND_ARMS: moving to pickup_xy=%s pickup_yaw=%.3f", pickup_xy, self._pickup_yaw())
            action, ready = self.move_controller.move_to_pose(
                sim_node,
                obs,
                action,
                pickup_xy,
                target_yaw=self._pickup_yaw(),
                position_thresh=0.025,
                yaw_thresh=0.035,
            )
            logger.info("EXTEND_ARMS: move_to_pose ready=%s", ready)
            if ready and self._pickup_alignment_ready(sim_node):
                self._set_extend_arm_targets(sim_node)
            return action, False

        done = sim_node.checkActionDone()
        if not done:
            logger.info("EXTEND_ARMS: left_target=%s left_current=%s right_target=%s right_current=%s",
                        sim_node.lft_arm_target_pose[:3], sim_node.sensor_lftarm_ep[:3],
                        sim_node.rgt_arm_target_pose[:3], sim_node.sensor_rgtarm_ep[:3])
        if done:
            logger.info("EXTEND_ARMS state completed")
        return action, done

    def _grasp_state_enter(self, sim_node):
        box_pos = self.box_original_pos
        # grasp_inset = 0.015
        # side_offset = max(0.5 * float(self.box_size[1]) - grasp_inset, 0.0)
        # top_offset = 0.0
        # target_posi1 = box_pos + np.array([0.0, side_offset, top_offset])
        # target_posi2 = box_pos + np.array([0.0, -side_offset, top_offset])
        side_offset = self._box_side_offset(0)
        front_offset = 0.03
        top_offset = self._box_top_offset(-0.01)
        target_posi1 = box_pos + np.array([front_offset, side_offset, top_offset])
        target_posi2 = box_pos + np.array([front_offset, -side_offset, top_offset])

        sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi1)
        sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi2)

        logger.info("GRASP: box_center=%s, box_size=%s, target_posi1=%s, target_posi2=%s, lft_arm_target_pose=%s, rgt_arm_target_pose=%s",
            box_pos, self.box_size, target_posi1, target_posi2, sim_node.lft_arm_target_pose, sim_node.rgt_arm_target_pose)

        sim_node.setArmEndTarget(
            sim_node.lft_arm_target_pose,
            sim_node.arm_action,
            "l",
            sim_node.sensor_lft_arm_qpos,
            self.lft_arm_rot,
        )
        sim_node.setArmEndTarget(
            sim_node.rgt_arm_target_pose,
            sim_node.arm_action,
            "r",
            sim_node.sensor_rgt_arm_qpos,
            self.rgt_arm_rot,
        )

    def _grasp_state_execute(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        left_close = np.allclose(sim_node.lft_arm_target_pose, sim_node.sensor_lftarm_ep, atol=3e-2)
        right_close = np.allclose(sim_node.rgt_arm_target_pose, sim_node.sensor_rgtarm_ep, atol=3e-2)
        done = left_close and right_close

        logger.info("GRASP: lft_arm_target_pose=%s, sensor_lftarm_ep=%s, rgt_arm_target_pose=%s, sensor_rgtarm_ep=%s",
                    sim_node.lft_arm_target_pose[:3], sim_node.sensor_lftarm_ep[:3],
                    sim_node.rgt_arm_target_pose[:3], sim_node.sensor_rgtarm_ep[:3])

        if done:
            sim_node.set_left_arm_new_target = False
            sim_node.set_right_arm_new_target = False
            logger.info("GRASP state completed: left_close=%s, right_close=%s", left_close, right_close)
        return action, done

    def _lift_state_enter(self, sim_node):
        self.state_data["lift_start_height"] = sim_node.sensor_slide_qpos[0]
        sim_node.tctr_slide[0] = self.state_data["lift_start_height"] - max(self.lift_height, 0.05)
        logger.info(
            "LIFT: start_height=%.4f, target_height=%.4f",
            self.state_data["lift_start_height"],
            sim_node.tctr_slide[0],
        )

    def _lift_state_execute(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        slide_done = np.allclose(sim_node.tctr_slide, sim_node.sensor_slide_qpos, atol=3e-2)
        if slide_done:
            logger.info("LIFT state completed: slide_height=%.4f", sim_node.sensor_slide_qpos[0])
        return action, slide_done

    def _backup_state_enter(self, sim_node):
        _, yaw = self.move_controller._get_robot_pose(sim_node)
        self.state_data["backup_xy"] = self._backup_clear_xy(sim_node)
        self.state_data["backup_yaw"] = yaw
        logger.info("BACKUP_CLEAR: backup_xy=%s, backup_yaw=%.3f", self.state_data["backup_xy"], yaw)

    def _backup_state_execute(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        action, done = self.move_controller.move_to_pose(
            sim_node,
            obs,
            action,
            self.state_data["backup_xy"],
            target_yaw=self.state_data["backup_yaw"],
            position_thresh=0.04,
            yaw_thresh=0.05,
            allow_reverse=True,
        )

        if done:
            logger.info("BACKUP_CLEAR state completed")
        return action, done

    def _adjust_height_state_enter(self, sim_node):
        current_hand_z = float(0.5 * (sim_node.sensor_lftarm_ep[2] + sim_node.sensor_rgtarm_ep[2]))
        target_hand_z = self._release_hand_z()

        # The MMK2 slide joint axis points downward. With the arm joints held,
        # increasing slide qpos lowers both grippers nearly one-to-one in world z.
        slide_height = float(sim_node.sensor_slide_qpos[0] + current_hand_z - target_hand_z)
        slide_min, slide_max = self._slide_ctrl_range(sim_node)
        slide_height = float(np.clip(slide_height, slide_min, slide_max))

        sim_node.tctr_slide[0] = slide_height
        self.state_data["adjust_target_hand_z"] = target_hand_z
        logger.info(
            "ADJUST_HEIGHT: current_hand_z=%.4f, target_hand_z=%.4f, slide_height=%.4f",
            current_hand_z,
            target_hand_z,
            slide_height,
        )

    def _adjust_height_state_execute(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        slide_done = (
            np.allclose(sim_node.tctr_slide, sim_node.sensor_slide_qpos, atol=2e-2)
            and np.abs(sim_node.sensor_slide_qvel).sum() < 1e-2
        )
        target_hand_z = self.state_data.get("adjust_target_hand_z", self._release_hand_z())
        hand_z = float(0.5 * (sim_node.sensor_lftarm_ep[2] + sim_node.sensor_rgtarm_ep[2]))
        hand_done = abs(hand_z - target_hand_z) < 0.03
        done = slide_done and hand_done
        if done:
            logger.info("ADJUST_HEIGHT state completed: hand_z=%.4f", hand_z)
        return action, done

    def _turn_around_state_enter(self, sim_node):
        self.state_data["turn_phase"] = "lane_turn"
        self.state_data["target_xy"] = self.box_target_pos[:2] + np.array([1.0, 0.0])  # 1 meter in front
        self.state_data["turn_yaw"] = self._yaw_from_to(self.state_data["target_xy"], self.box_target_pos[:2])
        logger.info("TURN_AROUND: target_xy=%s, turn_yaw=%.3f", self.state_data["target_xy"], self.state_data["turn_yaw"])

    def _turn_around_state_execute(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        phase = self.state_data.get("turn_phase", "lane_turn")

        if phase == "lane_turn":
            # Turn to face along y-axis (perpendicular to x)
            robot_pos, _ = self.move_controller._get_robot_pose(sim_node, obs)
            target_xy = self.state_data["target_xy"]
            # If robot y > target y, face negative-y; else face positive-y
            if robot_pos[1] > target_xy[1]:
                lane_yaw = -3.14159 / 2  # face negative-y
            else:
                lane_yaw = 3.14159 / 2   # face positive-y

            action, turn_done = self.move_controller.turn_to_yaw(
                sim_node,
                obs,
                action,
                lane_yaw,
                yaw_thresh=0.10,
                min_yaw_rate=0.60,
            )
            if turn_done:
                self.state_data["turn_phase"] = "lane_drive"
                self.state_data["lane_yaw"] = lane_yaw
                self.move_controller.stop(sim_node, action)
                robot_pos, yaw = self.move_controller._get_robot_pose(sim_node, obs)
                logger.info("TURN_AROUND: lane_turn completed, lane_yaw=%.3f, pos=%s", lane_yaw, robot_pos)
            return action, False

        if phase == "lane_drive":
            robot_pos, yaw = self.move_controller._get_robot_pose(sim_node, obs)
            target_xy = self.state_data["target_xy"]
            y_err = float(target_xy[1] - robot_pos[1])
            lane_yaw = self.state_data["lane_yaw"]
            yaw_err = self.move_controller._wrap_angle(lane_yaw - yaw)

            if abs(y_err) < 0.025:
                self.state_data["turn_phase"] = "yaw"
                self.move_controller.stop(sim_node, action)
                logger.info("TURN_AROUND: lane_drive completed at pos=%s y_err=%.3f", robot_pos, y_err)
                return action, False

            # Move forward with heading correction (like ALIGN_Y)
            v = np.clip(0.65 * abs(y_err), 0.10, 0.38)
            action = self.move_controller.drive_velocity(sim_node, action, v, 2.5 * yaw_err)
            return action, False

        if phase == "yaw":
            # Turn to face the target (shelf)
            action, done = self.move_controller.turn_to_yaw(
                sim_node,
                obs,
                action,
                self.state_data["turn_yaw"],
                yaw_thresh=0.04,
                min_yaw_rate=0.60,
            )
            if done:
                logger.info("TURN_AROUND state completed")
            return action, done

        return action, False

    def _move_to_target_state_enter(self, sim_node):
        self.state_data["move_to_target_phase"] = "move_xy"
        logger.info("MOVE_TO_TARGET: target=%s", self.box_target_pos[:2])

    def _move_to_target_state_execute(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        phase = self.state_data.get("move_to_target_phase", "move_xy")

        if phase == "move_xy":
            box_pos = self._current_box_pos(sim_node)
            box_x = float(box_pos[0])
            x_err = float(self.box_target_pos[0] - box_x)

            logger.info("MOVE_TO_TARGET: box=%s, target_x=%.3f, x_err=%.3f",
                        box_pos, self.box_target_pos[0], x_err)

            if abs(x_err) < 0.03:
                self.move_controller.stop(sim_node, action)
                logger.info("MOVE_TO_TARGET: x-y reached, switching to DESCEND state")
                return action, True  # 切换到DESCEND状态

            # Determine direction: if box is in front of target (box_x > target_x), move backward; else move forward
            if box_x > self.box_target_pos[0]:
                v = np.clip(0.5 * abs(x_err), 0.05, 0.15)  # move backward (positive-x)
            else:
                v = -np.clip(0.5 * abs(x_err), 0.05, 0.15)  # move forward (negative-x)
            action = self.move_controller.drive_velocity(sim_node, action, v, 0.0)
            return action, False

        return action, True

    def _descend_state_enter(self, sim_node):
        # Calculate target slide height for descent
        # target_hand_z = 放置位置 + 箱子高度一半（手在箱子底部）
        target_hand_z = self.box_target_pos[2] + 0.5 * self.box_size[0]
        # 当前手部高度
        current_hand_z = float(0.5 * (sim_node.sensor_lftarm_ep[2] + sim_node.sensor_rgtarm_ep[2]))
        # tctr_slide[0] 值越小高度越高，所以下降时 slide 值需要增加
        # slide变化量 = 当前手部高度 - 目标手部高度（正值表示需要下降/升高slide）
        slide_delta = current_hand_z - target_hand_z
        slide_height = float(sim_node.sensor_slide_qpos[0] + slide_delta)
        slide_min, slide_max = self._slide_ctrl_range(sim_node)
        sim_node.tctr_slide[0] = float(np.clip(slide_height, slide_min, slide_max))
        logger.info("DESCEND: slide_height=%.4f (current_slide=%.4f, current_hand_z=%.4f, target_hand_z=%.4f, delta=%.4f)",
                   sim_node.tctr_slide[0], sim_node.sensor_slide_qpos[0], current_hand_z, target_hand_z, slide_delta)

    def _descend_state_execute(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        # Wait for slide to reach target height
        slide_done = (
            np.allclose(sim_node.tctr_slide, sim_node.sensor_slide_qpos, atol=2e-2)
            and np.abs(sim_node.sensor_slide_qvel).sum() < 1e-2
        )
        if slide_done:
            self.move_controller.stop(sim_node, action)
            hand_z = float(0.5 * (sim_node.sensor_lftarm_ep[2] + sim_node.sensor_rgtarm_ep[2]))
            logger.info("DESCEND: descent completed, hand_z=%.4f, slide=%.4f", hand_z, sim_node.sensor_slide_qpos[0])
            return action, True
        return action, False

    def _release_state_enter(self, sim_node):
        self.state_data["release_phase"] = "open_arms"
        # Get current arm endpoint positions
        lft_arm_pos = sim_node.sensor_lftarm_ep.copy()
        rgt_arm_pos = sim_node.sensor_rgtarm_ep.copy()

        # Move arms outward 0.05m and backward 0.1m
        target_posi1 = lft_arm_pos + np.array([-0.1, 0.08, 0.0])
        target_posi2 = rgt_arm_pos + np.array([-0.1, -0.08, 0.0])

        sim_node.lft_arm_target_pose[:] = target_posi1
        sim_node.rgt_arm_target_pose[:] = target_posi2

        sim_node.setArmEndTarget(
            sim_node.lft_arm_target_pose,
            sim_node.arm_action,
            "l",
            sim_node.sensor_lft_arm_qpos,
            self.lft_arm_rot,
        )
        sim_node.setArmEndTarget(
            sim_node.rgt_arm_target_pose,
            sim_node.arm_action,
            "r",
            sim_node.sensor_rgt_arm_qpos,
            self.rgt_arm_rot,
        )

        # Open grippers to release box
        sim_node.tctr_lft_gripper[:] = 0.0
        sim_node.tctr_rgt_gripper[:] = 0.0

        logger.info("RELEASE: lft=%s->%s, rgt=%s->%s", lft_arm_pos, target_posi1, rgt_arm_pos, target_posi2)

    def _release_state_execute(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        phase = self.state_data.get("release_phase", "open_arms")

        if phase == "open_arms":
            left_close = np.allclose(sim_node.lft_arm_target_pose, sim_node.sensor_lftarm_ep, atol=5e-2)
            right_close = np.allclose(sim_node.rgt_arm_target_pose, sim_node.sensor_rgtarm_ep, atol=5e-2)
            gripper_done = (
                np.allclose(sim_node.tctr_lft_gripper, sim_node.sensor_lft_gripper_qpos, atol=0.05)
                and np.allclose(sim_node.tctr_rgt_gripper, sim_node.sensor_rgt_gripper_qpos, atol=0.05)
            )
            if left_close and right_close and gripper_done:
                self.state_data["release_phase"] = "backup"
                self.state_data["backup_start_x"] = sim_node.mj_data.qpos[0]
                self.move_controller.stop(sim_node, action)
                logger.info("RELEASE: arms opened, starting backup")
            return action, False

        if phase == "backup":
            robot_pos, robot_yaw = self.move_controller._get_robot_pose(sim_node, obs)
            start_x = self.state_data.get("backup_start_x", robot_pos[0])
            backup_dist = 0.2
            target_x = start_x + backup_dist

            action, done = self.move_controller.move_to_pose(
                sim_node,
                obs,
                action,
                np.array([target_x, robot_pos[1]]),
                target_yaw=robot_yaw,
                position_thresh=0.04,
                yaw_thresh=0.05,
                allow_reverse=True,
            )

            if done:
                self.move_controller.stop(sim_node, action)
                sim_node.set_left_arm_new_target = False
                sim_node.set_right_arm_new_target = False
                logger.info("RELEASE state completed")
                return action, True
            return action, False

        return action, False

    def execute_step(self, sim_node, obs, action) -> Tuple[np.ndarray, bool]:
        if self.is_complete():
            self.move_controller.stop(sim_node, action)
            return action, True

        state = self.state_machine.state_idx
        if self.state_machine.trigger():
            self._enter_state(sim_node, action)
            dif = np.abs(action - sim_node.target_control)
            sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)
            if sim_node.joint_move_ratio.shape[0] > 2:
                sim_node.joint_move_ratio[2] *= 0.25

        action, state_done = self._execute_state_logic(sim_node, obs, action)

        if not self._state_uses_base_motion(state):
            self._hold_base_pose(sim_node, obs, action)

        for i in range(2, sim_node.njctrl):
            action[i] = step_func(
                action[i],
                sim_node.target_control[i],
                self.move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t,
            )

        if state_done:
            self.move_controller.stop(sim_node, action)
            self.state_machine.next()
        else:
            self.state_machine.update()

        return action, self.is_complete()

    def get_current_state(self) -> int:
        return self.state_machine.state_idx

    def get_state_name(self, state_id: Optional[int] = None) -> str:
        if state_id is None:
            state_id = self.state_machine.state_idx

        state_names = {
            self.STATE_INIT: "INIT",
            self.STATE_ALIGN_Y: "ALIGN_Y",
            self.STATE_APPROACH: "APPROACH",
            self.STATE_EXTEND_ARMS: "EXTEND_ARMS",
            self.STATE_GRASP: "GRASP",
            self.STATE_LIFT: "LIFT",
            self.STATE_TURN_AROUND: "TURN_AROUND",
            self.STATE_ADJUST_HEIGHT: "ADJUST_HEIGHT",
            self.STATE_BACKUP: "BACKUP_CLEAR",
            self.STATE_MOVE_TO_TARGET: "MOVE_TO_TARGET",
            self.STATE_DESCEND: "DESCEND",
            self.STATE_RELEASE: "RELEASE",
        }
        return state_names.get(state_id, "UNKNOWN")
