"""
搬箱子功能使用示例

演示如何使用 BoxTransport 类实现机器人自动搬运箱子
从栈板（右侧）搬运到线棒车（左侧），每层并排放三个
"""
from time import sleep

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import argparse
import logging
import cv2

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
import multiprocessing as mp

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase, recoder_mmk2, copypy2, BoxTransport
from discoverse.utils import get_body_tmat


class SimNode(MMK2TaskBase):
    """仿真节点类"""

    def post_load_mjcf(self):
        """保存箱子初始位置"""
        super().post_load_mjcf()
        self.box_initial_qpos = {}
        for name in self.free_body_qpos_ids.keys():
            if "_box_" in name:
                qid = self.mj_model.jnt_qposadr[self.free_body_qpos_ids[name]]
                self.box_initial_qpos[name] = self.mj_data.qpos[qid:qid+7].copy()

    def resetState(self):
        """重置状态，保留箱子初始位置"""
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:self.njq] = self.init_joint_pose[:]
        self.mj_data.ctrl[:self.njctrl] = self.init_joint_ctrl[:]
        # 恢复箱子初始位置
        for name, qpos in self.box_initial_qpos.items():
            qid = self.mj_model.jnt_qposadr[self.free_body_qpos_ids[name]]
            self.mj_data.qpos[qid:qid+7] = qpos[:]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def domain_randomization(self):
        """域随机化"""
        # 随机机器人初始位置
        self.mj_data.qpos[self.njq+0] += (np.random.random() - 0.5) * 0.1
        self.mj_data.qpos[self.njq+1] += (np.random.random() - 0.5) * 0.1
        self.mj_data.qpos[self.njq+2] += (np.random.random() - 0.5) * 0.3

    def check_success(self):
        """检查任务是否成功"""
        # 检查是否所有箱子都已搬运到线棒车上
        return True  # 暂时简化


def get_shelf_positions():
    """获取线棒车上的放置位置（从左到右，从上到下）"""
    positions = []
    # 线棒车宽度500mm，x范围约-0.25到0.25
    # 长度方向：-0.6, 0, 0.6 (沿y轴，间隔400mm，从左到右)
    y_positions = [-0.6, 0.0, 0.6]
    # 3层架子，高度分别为1.00m, 0.65m, 0.30m（从上到下）
    heights = [1.00, 0.65, 0.30]

    # 从左到右，从上到下：先遍历y（列），再遍历height（每列内从高到低）
    for y in y_positions:
        for h in heights:
            positions.append(np.array([-1.5, y, h]))

    return positions


def get_pallet_boxes_from_xml(sim_node):
    """从XML中读取栈板上所有箱子的名称、位置和尺寸"""
    boxes = []
    box_size = None

    # 收集所有箱子的body name
    box_names = []
    for i in range(sim_node.mj_model.nbody):
        body_name = sim_node.mj_model.body(i).name
        if body_name and "_box_" in body_name:
            box_names.append(body_name)

    logger.info(f"Found {len(box_names)} box bodies in model")

    # 按y坐标从左到右排序，再按z坐标从上到下排序
    box_positions = []
    for name in box_names:
        try:
            # 使用 qpos 直接获取位置（箱子是 free joint）
            qid = sim_node.free_body_qpos_ids.get(name)
            if qid is None:
                logger.warning(f"Box {name} not in free_body_qpos_ids")
                continue
            # free_body_qpos_ids 存的是 joint ID，需要用 jnt_qposadr 转换
            qpos_adr = sim_node.mj_model.jnt_qposadr[qid]
            pos = sim_node.mj_data.qpos[qpos_adr:qpos_adr+3].copy()
            logger.info(f"Box {name} at position {pos}")

            # 获取箱子尺寸 - 遍历所有geom找到属于这个body的
            if box_size is None:
                for i in range(sim_node.mj_model.ngeom):
                    body_id = sim_node.mj_model.geom_bodyid[i]
                    body_name_of_geom = sim_node.mj_model.body(body_id).name
                    if body_name_of_geom == name:
                        geom_size = sim_node.mj_model.geom_size[i]
                        box_size = np.array(geom_size) * 2  # 转换为完整尺寸
                        logger.info(f"Box size from geom: {box_size}")
                        break

            box_positions.append({"name": name, "pos": pos})
        except Exception as e:
            logger.warning(f"Error processing box {name}: {e}")
            continue

    # 排序：先按y取整分组(从左到右)，组内再按z(从上到下)
    box_positions.sort(key=lambda x: (round(x["pos"][1], 1), -x["pos"][2]))
    logger.info("Sorted box order: %s", [(b['name'], round(b['pos'][1], 1), b['pos'][2]) for b in box_positions])

    # 分配编号
    for idx, box_info in enumerate(box_positions, start=1):
        boxes.append({
            "name": box_info["name"],
            "label": str(idx),
            "original_pos": box_info["pos"]
        })

    return boxes, box_size


def main():
    """主函数"""
    # 配置参数
    cfg = MMK2Cfg()
    cfg.use_gaussian_renderer = False
    cfg.gs_model_dict["box"] = "object/box.ply"
    cfg.gs_model_dict["background"] = "scene/tsimf_library_0/point_cloud_for_mmk2.ply"
    cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_box_demo.xml"
    cfg.sync = True
    cfg.headless = False
    cfg.render_set = {
        "fps": 10,
        "width": 640,
        "height": 480
    }
    cfg.obs_rgb_cam_id = [0, 1, 2]
    cfg.save_mjb_and_task_config = True
    # 设置机器人初始位置在线棒车和栈板之间
    cfg.init_state["base_position"] = [0.0, 0.0, 0.0]

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    parser.add_argument('--use_gs', action='store_true', help='Use gaussian splatting renderer')
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False
    cfg.use_gaussian_renderer = args.use_gs

    # 创建保存目录
    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/mmk2_box_transport")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建仿真节点
    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    # 设置相机看向场景中心
    sim_node.free_camera.lookat = np.array([1.5, 0.0, 0.5])
    sim_node.free_camera.distance = 5.0
    sim_node.free_camera.elevation = 30
    sim_node.free_camera.azimuth = 45

    # 获取线棒车和栈板位置
    tmat_shelf = get_body_tmat(sim_node.mj_data, "shelf_frame")
    logger.info(f"Shelf position: {tmat_shelf[:3, 3]}")

    # 初始化动作
    action = np.zeros_like(sim_node.target_control)
    process_list = []

    # 仿真参数
    max_time = 2000.0  # 最大仿真时间（秒）
    move_speed = 2.0

    # 重置仿真
    obs = sim_node.reset()

    # 稳定物理引擎，等速度接近0
    for _ in range(200):
        sim_node.step(action)
    while np.abs(sim_node.mj_data.qvel).sum() > 0.1:
        logger.info(f"Waiting for velocity to settle... {np.abs(sim_node.mj_data.qvel).sum()}")
        sim_node.step(action)

    # 获取所有箱子和放置位置
    pallet_boxes, box_size = get_pallet_boxes_from_xml(sim_node)
    shelf_positions = get_shelf_positions()

    # 默认箱子尺寸（如果无法从XML获取）
    if box_size is None:
        logger.error("Could not get box size from XML, using default")
        return

    logger.info(f"Found {len(pallet_boxes)} boxes on pallets, size={box_size.tolist()}")
    logger.info(f"Shelf has {len(shelf_positions)} positions")
    task_count = min(len(pallet_boxes), len(shelf_positions))
    if task_count < len(pallet_boxes):
        logger.warning("Only %d shelf positions available for %d boxes; extra boxes will be skipped", task_count, len(pallet_boxes))

    # 数据记录
    act_lst, obs_lst = [], []

    # 当前正在搬运的箱子索引和BoxTransport实例
    current_box_idx = 0
    box_transport = None

    # 主循环
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            if box_transport:
                box_transport.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []

            obs = sim_node.getObservation()
            continue

        # 如果还没有创建BoxTransport或者当前箱子已经搬运完成
        if box_transport is None or box_transport.is_complete():
            if current_box_idx >= task_count:
                # 所有箱子都搬运完成
                logger.info("All boxes transferred!")
                break

            # 获取当前箱子和目标位置
            box_info = pallet_boxes[current_box_idx]
            target_pos = shelf_positions[current_box_idx]

            logger.info(f"Box {current_box_idx}/{len(pallet_boxes)} [{box_info.get('label', '')}]: {box_info['name']} from {box_info['original_pos']} to {target_pos}")

            # 创建新的搬箱子任务
            box_transport = BoxTransport(
                box=box_info,
                box_target_pos=target_pos,
                box_size=box_size,
                approach_distance=0.95,
                lift_height=0.2,
                move_speed=move_speed,
                move_init_speed=8.0
            )

            current_box_idx += 1

        try:
            # 执行搬箱子步骤
            action, _ = box_transport.execute_step(sim_node, obs, action)

            # 检查超时
            if sim_node.mj_data.time > max_time:
                raise ValueError("Time out")

        except Exception as e:
            logger.error(f"Exception caught: {e}", e)
            # sleep(int())
            break

        # 执行仿真步
        obs, _, _, _, _ = sim_node.step(action)

        # 显示视频窗口 - 三个视频垂直拼接在一个窗口中
        show_video = False
        if show_video and not cfg.headless and "img" in obs:
            images = []
            labels = ["FPV", "Left Arm", "Right Arm"]
            for i in range(3):
                if i in obs["img"] and obs["img"][i] is not None:
                    img = obs["img"][i].copy()
                    # 添加标签
                    cv2.putText(img, labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    images.append(img)

            if images:
                # 垂直拼接
                combined = np.vstack(images)
                cv2.imshow("Robot View", combined)
                cv2.waitKey(1)

        # 记录数据
        robot_pos = sim_node.mj_data.qpos[:3]
        box_pos = pallet_boxes[current_box_idx-1]["original_pos"]
        dist_to_box = float(np.linalg.norm(robot_pos[:2] - box_pos[:2]))
        # logger.info(f"Box {current_box_idx}/{len(pallet_boxes)} | State: {box_transport.get_state_name()} | "
        #            f"box={box_pos[:2]} target={box_transport.box_target_pos[:2]} robot={robot_pos[:2]} dist_to_box={dist_to_box:.3f} | "
        #            f"action=[{action[0]:.2f}, {action[1]:.2f}] | "
        #            f"slide_height tctr={sim_node.tctr_slide[0]:.4f} sensor={sim_node.sensor_slide_qpos[0]:.4f}")

        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)

    # 保存最后一条数据
    if act_lst and obs_lst:
        save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
        process = mp.Process(target=recoder_mmk2, args=(save_path, act_lst, obs_lst, cfg))
        process.start()
        process_list.append(process)
        data_idx += 1

    # 等待所有进程完成
    for p in process_list:
        p.join()

    cv2.destroyAllWindows()
    logger.info("Simulation ended")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    main()
