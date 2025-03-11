import cv2
from ultralytics import YOLO
import time
import threading
from pymavlink import mavutil
from queue import Queue


model = YOLO(r"runs\tunnelsacn.pt")

# 初始化摄像头
cap = cv2.VideoCapture(0)  

# MAVLink连接配置
MAVLINK_CONNECTION = 'udp:localhost' 
position_queue = Queue(maxsize=1)  

# 检测参数配置
min_width = 300
min_height = 300
log_file = 'detection_log.txt'
report_file = 'detection_report.txt'

# 状态跟踪变量
start_time = time.time()
total_detections = 0
alert_count = 0
max_width = 0
max_height = 0
class_counts = {}
alert_events = []

# 处理频率控制
PROCESS_FPS = 2  # 每秒处理帧数
frame_interval = 1.0 / PROCESS_FPS
last_process_time = time.time()

def mavlink_listener():
    """MAVLink位置监听线程"""
    mav = mavutil.mavlink_connection(MAVLINK_CONNECTION)
    
    print("等待MAVLink心跳...")
    mav.wait_heartbeat()
    print("MAVLink连接成功!")
    
    while True:
        try:
            msg = mav.recv_match(blocking=True, timeout=5)
            if not msg:
                continue
                
            if msg.get_type() == 'LOCAL_POSITION_NED':
                # 获取相对位置（相对于起飞点）
                position = (msg.x, msg.y, msg.z)
                # 更新位置队列（只保留最新位置）
                if position_queue.full():
                    position_queue.get()
                position_queue.put(position)
                
        except Exception as e:
            print(f"MAVLink错误: {str(e)}")
            break

def get_drone_position():
    """从队列获取最新无人机位置"""
    try:
        return position_queue.get_nowait()
    except:
        return (0.0, 0.0, 0.0)  # 默认位置

def write_log(data):
    """写入日志文件"""
    with open(log_file, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {data}\n")

def generate_report():
    """生成检测报告"""
    with open(report_file, 'w') as f:
        f.write("======== 无人机视觉检测报告 ========\n\n")
        f.write(f"视频分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总检测次数: {total_detections}\n")
        f.write(f"触发警报次数: {alert_count}\n")
        f.write(f"最大检测尺寸: {max_width:.1f}x{max_height:.1f}px\n\n")

        f.write("目标类别分布:\n")
        for cls, count in class_counts.items():
            f.write(f"- {cls}: {count}次\n")

        f.write("\n警报事件时间线:\n")
        for event in alert_events:
            f.write(f"[{event['time']}] {event['class']} ({event['width']:.1f}x{event['height']:.1f}px) "
                    f"@ (x={event['position'][0]:.1f}, y={event['position'][1]:.1f}, z={event['position'][2]:.1f})\n")

# 启动MAVLink监听线程
mav_thread = threading.Thread(target=mavlink_listener, daemon=True)
mav_thread.start()

# 主循环
try:
    while cap.isOpened():
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败")
            break

        # 控制处理频率
        current_time = time.time()
        if current_time - last_process_time < frame_interval:
            continue
            
        last_process_time = current_time

        # 执行目标检测
        results = model(frame)

        # 处理检测结果
        for result in results:
            for box in result.boxes:
                total_detections += 1
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                xyxy = box.xyxy[0].tolist()
                width = xyxy[2] - xyxy[0]
                height = xyxy[3] - xyxy[1]

                # 更新统计数据
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                max_width = max(max_width, width)
                max_height = max(max_height, height)

                # 写入日志
                log_entry = f"{cls_name} ({width:.1f}x{height:.1f}px) @ [{xyxy[0]:.1f},{xyxy[1]:.1f}]"
                write_log(log_entry)

                # 触发警报检测
                if width > min_width and height > min_height:
                    alert_count += 1
                    position = get_drone_position()
                    alert_time = time.strftime('%Y-%m-%d %H:%M:%S')
                    alert_events.append({
                        'time': alert_time,
                        'class': cls_name,
                        'width': width,
                        'height': height,
                        'position': position
                    })
                    geo_log = f"ALERT: {cls_name} @ (x={position[0]:.1f}, y={position[1]:.1f}, z={position[2]:.1f})"
                    write_log(geo_log)
                    print(f"[警报] {geo_log}")

        # 显示实时画面（可选）
        cv2.imshow('Drone Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("用户中断操作")

finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    generate_report()
    print("实时检测结束，检测报告已生成")
