# Task 3.19 Video Evidence

## 1. 完整多物体演示视频

GitHub 提交版视频，完整时长、压缩到 480p/10fps：

```text
docs/task319_delivery/videos/task319_system_demo_480p.mp4
```

原始 720p 本地证据视频：

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100/grasp8.mp4
```

原始视频大小约 71 MB。配套结果目录：

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100/
```

配套压缩证据：

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100/mind_sort_demo.zip
```

说明：

- `video_manifest.json` 中仍记录原始视频名 `external_grasp_demo.mp4`。
- 当前目录实际保留的视频文件名是 `grasp8.mp4`。
- GitHub 提交版视频由 `grasp8.mp4` 压缩生成，展示同一条 8 个物体识别、分类、导航到桶、桶口释放流程。
- 该 run 启用了显式辅助抓取路径，不能作为严格物理夹爪全成功证明。

## 2. 最新严格物理单轮视频

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260626_145846/external_grasp_demo.mp4
```

配套结果：

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260626_145846/mind_sort_demo/mind_sort_task_queue.json
```

结果摘要：

- 目标：`trash_potted_meat_can_0`
- 分类：厨余垃圾
- 严格物理抓取：失败
- 原因：cuRobo Cartesian final descent 未到达目标/近距阈值，未启用辅助 attach/carry。

## 3. 重新录制命令

完整演示视频：

```bash
cd mda_isaaclab
python task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --mind_sort_demo \
  --mind_sort_suction_assist \
  --no-mind_sort_gripper_proximity_assist \
  --mind_sort_allow_stale_reshoot_for_suction_demo \
  --no-target_reachability_ik_check \
  --record_video \
  --video_width 1280 \
  --video_height 720 \
  --video_sample_stride 4 \
  --no-gui_realtime_playback
```

严格物理单轮：

```bash
cd mda_isaaclab
python task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --mind_sort_demo \
  --mind_sort_max_objects 1 \
  --no-mind_sort_gripper_proximity_assist \
  --no-mind_sort_suction_assist \
  --record_video \
  --video_width 1280 \
  --video_height 720 \
  --video_sample_stride 4 \
  --no-gui_realtime_playback
```

录制完成后查看：

```text
task_319_garbage_sort/output/head_camera_grasp_records/<timestamp>/video_manifest.json
```
