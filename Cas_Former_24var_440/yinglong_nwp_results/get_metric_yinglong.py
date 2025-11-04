import os
import h5py
import numpy as np
from datetime import date, timedelta
from typing import List, Tuple, Optional

# 配置路径
FORECAST_ROOT = "/nfs/samba/数据聚变/气象数据/hefang/upper_hrrr+sfc_hrrr+same_resolution/test/forecast_result_2024/yinglong_modelww_12_inference_48h_pangu_marge_with_slbc/2024"
TRUTH_ROOT = "/nfs/samba/数据聚变/气象数据/hrrr_440_30var/test/2024"
OUTPUT_DIR = "/home/pengjian/CAS_Former_440/yinglong_nwp_results"
OUTPUT_FILE = "invalid_files_yinglong.txt"

# 预期的数据形状
FORECAST_SHAPE = (48, 24, 440, 408)
TRUTH_SHAPE = (24, 30, 440, 408)


def crop_hw(x: np.ndarray, pad: int = 64) -> np.ndarray:
    """裁剪数据的高和宽
    
    Args:
        x: 输入数据
        pad: 裁剪的边界大小
        
    Returns:
        裁剪后的数据
    """
    return x[..., pad:-pad, pad:-pad] if pad > 0 else x


def day_file(root: str, d: date) -> str:
    """生成日期对应的文件路径
    
    Args:
        root: 根目录
        d: 日期
        
    Returns:
        文件路径
    """
    return os.path.join(root, f"{d:%m}", f"{d:%d}.h5")


def check_data_format(file_path: str, expected_shape: Tuple[int, ...], data_name: str) -> Tuple[bool, str]:
    """检查数据格式是否正确
    
    Args:
        file_path: 文件路径
        expected_shape: 期望的数据形状
        data_name: 数据名称（用于错误信息）
        
    Returns:
        (是否有效, 错误信息)
    """
    try:
        with h5py.File(file_path, "r") as hf:
            if "fields" not in hf:
                return False, f"缺少'fields'字段"
            data = hf["fields"][:]
            if data.shape != expected_shape:
                return False, f"形状不匹配: 期望{expected_shape}, 实际{data.shape}"
            return True, "OK"
    except Exception as e:
        return False, f"读取错误: {str(e)}"


def check_file(file_path: str, expected_shape: Tuple[int, ...], file_type: str, 
               invalid_files: List[str]) -> bool:
    """检查文件是否存在且格式正确
    
    Args:
        file_path: 文件路径
        expected_shape: 期望的数据形状
        file_type: 文件类型（用于错误信息）
        invalid_files: 无效文件列表
        
    Returns:
        文件是否有效
    """
    if not os.path.exists(file_path):
        invalid_files.append(f"{file_type}文件不存在: {file_path}")
        return False
    
    is_valid, msg = check_data_format(file_path, expected_shape, file_type)
    if not is_valid:
        invalid_files.append(f"{file_type}数据格式错误: {file_path} - {msg}")
        return False
    
    return True


def validate_date(d: date, invalid_files: List[str]) -> bool:
    """验证特定日期的所有相关文件
    
    Args:
        d: 要验证的日期
        invalid_files: 无效文件列表
        
    Returns:
        日期是否有效（所有文件都存在且格式正确）
    """
    # 获取所有需要检查的文件路径
    forecast_file = day_file(FORECAST_ROOT, d)
    truth_today_file = day_file(TRUTH_ROOT, d)
    truth_next_day_file = day_file(TRUTH_ROOT, d + timedelta(days=1))
    truth_day_after_next_file = day_file(TRUTH_ROOT, d + timedelta(days=2))
    
    # 检查预测文件
    if not check_file(forecast_file, FORECAST_SHAPE, "预测", invalid_files):
        return False
    
    # 检查当天真值文件
    if not check_file(truth_today_file, TRUTH_SHAPE, "当天真值", invalid_files):
        return False
    
    # 检查下一天真值文件
    if not check_file(truth_next_day_file, TRUTH_SHAPE, "下一天真值", invalid_files):
        return False
    
    # 检查后一天真值文件
    if not check_file(truth_day_after_next_file, TRUTH_SHAPE, "后一天真值", invalid_files):
        return False
    
    return True


def main():
    """主函数"""
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化变量
    invalid_files: List[str] = []
    valid_days: List[date] = []
    
    # 遍历2024年的每一天
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    
    current_date = start_date
    while current_date <= end_date:
        if validate_date(current_date, invalid_files):
            valid_days.append(current_date)
        current_date += timedelta(days=1)
    
    # 输出结果
    print(f"有效天数: {len(valid_days)}")
    print(f"yinglong有效日期:{valid_days}")
    print(f"异常文件数量: {len(invalid_files)}")
    
    # 输出异常文件信息
    if invalid_files:
        print("\n异常文件列表:")
        for i, msg in enumerate(invalid_files, 1):
            print(f"{i:3d}. {msg}")
            
        # 保存异常文件列表
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        with open(output_path, "w", encoding="utf-8") as f:
            for msg in invalid_files:
                f.write(msg + "\n")
        print(f"异常文件列表已保存到: {output_path}")


if __name__ == "__main__":
    main()