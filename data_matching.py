"""
data_matching.py - 修复版
数据匹配模块：负责扫描、匹配和绑定RGB与多光谱图像样本
"""

import os
import re
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ===================== 配置参数 =====================
DEFAULT_CONFIG = {
    'DATASET_ROOT': 'data-RGBMS',
    'CSV_OUTPUT_PATH': "rgbms_samples_list.csv",
    'MS_CHANNELS': ['G', 'NIR', 'R', 'RE'],
    'RGB_CHANNEL': 'RGB',
    'TARGET_SPLITS': ['train', 'val'],
    'IGNORE_SPLITS': ['test'],
    'SUPPORTED_EXTS': ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'],
    'MATCH_TOLERANCE': True,
    'FALLBACK_MATCH': True,
    'MISSING_CHANNEL_ALLOW': False
}

# ===================== 1. DJI特征提取器类 =====================
class DJIFileFeatureExtractor:
    """增强的DJI文件名特征提取器"""

    def __init__(self, file_path, config=None):
        self.file_path = file_path
        self.config = config or DEFAULT_CONFIG
        self.features = self._initialize_features()
        self._extract_all_features()

    def _initialize_features(self):
        """初始化特征字典"""
        return {
            'filename': os.path.basename(self.file_path),
            'base_name': os.path.splitext(os.path.basename(self.file_path))[0],
            'extension': os.path.splitext(self.file_path)[1].lower(),
            'file_size': os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0,
            'file_mtime': os.path.getmtime(self.file_path) if os.path.exists(self.file_path) else 0,
            'file_mtime_str': '',
            'is_image': False,
            'dji_prefix': False,
            'raw_timestamp': None,
            'sequence_number': None,
            'full_dji_code': None,
            'datetime_obj': None,
            'date_str': None,
            'time_str': None,
            'year': None,
            'month': None,
            'day': None,
            'channel_type': None,
            'row_num': None,
            'col_num': None,
            'row_col': None,
            'grid_id': None,
            'date_folder': None,
            'split_type': None,
            'parse_success': False,
            'parse_score': 0.0
        }

    def _extract_all_features(self):
        """执行所有特征提取步骤"""
        self._extract_basic_info()
        self._extract_dji_and_time()
        self._extract_channel_and_space()
        self._extract_path_info()
        self._calculate_parse_score()

    def _extract_basic_info(self):
        """提取文件基础信息"""
        if self.features['extension'] in self.config['SUPPORTED_EXTS']:
            self.features['is_image'] = True

        if self.features['file_mtime'] > 0:
            from datetime import datetime
            self.features['file_mtime_str'] = datetime.fromtimestamp(
                self.features['file_mtime']
            ).strftime("%Y-%m-%d %H:%M:%S")

    def _extract_dji_and_time(self):
        """提取DJI编码和时间信息"""
        base = self.features['base_name']
        dji_pattern = r'DJI[_-]?(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})[_-]?(\d{1,4})'
        match = re.search(dji_pattern, base, re.IGNORECASE)

        if match:
            self.features['dji_prefix'] = True
            year, month, day, hour, minute, second = match.groups()[:6]
            self.features['sequence_number'] = match.group(7).zfill(4)
            self.features['raw_timestamp'] = f"{year}{month}{day}{hour}{minute}{second}"
            self.features['full_dji_code'] = f"DJI_{self.features['raw_timestamp']}_{self.features['sequence_number']}"

            try:
                from datetime import datetime
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                self.features['datetime_obj'] = dt
                self.features['date_str'] = dt.strftime("%Y-%m-%d")
                self.features['time_str'] = dt.strftime("%H:%M:%S")
                self.features['year'] = year
                self.features['month'] = month
                self.features['day'] = day
            except:
                pass
        else:
            dji_loose = r'DJI[_-]?(\d+)_?(\d+)'
            match_loose = re.search(dji_loose, base, re.IGNORECASE)
            if match_loose:
                self.features['dji_prefix'] = True
                self.features['raw_timestamp'] = match_loose.group(1)
                self.features['sequence_number'] = match_loose.group(2).zfill(4)
                self.features['full_dji_code'] = f"DJI_{self.features['raw_timestamp']}_{self.features['sequence_number']}"

                ts = self.features['raw_timestamp']
                if len(ts) >= 8:
                    try:
                        from datetime import datetime
                        dt = datetime.strptime(ts[:8], "%Y%m%d")
                        self.features['datetime_obj'] = dt
                        self.features['date_str'] = dt.strftime("%Y-%m-d")
                        self.features['year'] = ts[:4]
                        self.features['month'] = ts[4:6]
                        self.features['day'] = ts[6:8]
                    except:
                        pass

    def _extract_channel_and_space(self):
        """提取通道和行列号信息"""
        base = self.features['base_name'].upper()

        channel_keywords = {
            self.config['RGB_CHANNEL'].upper(): ['RGB', '_RGB_', 'COLOR', 'RGB8'],
            'G': ['G', 'GREEN', 'GRN', '_G_'],
            'NIR': ['NIR', 'IR', '_NIR_', 'INFRARED'],
            'R': ['R', 'RED', '_R_'],
            'RE': ['RE', 'REDEDGE', '_RE_', 'RED_EDGE']
        }

        for channel, keywords in channel_keywords.items():
            if any(keyword in base for keyword in keywords):
                self.features['channel_type'] = channel
                break

        row_patterns = [
            r'ROW\s*[_-]?\s*(\d+)', r'行\s*[_-]?\s*(\d+)',
            r'R\s*[_-]?\s*(\d+)', r'(\d+)\s*行'
        ]
        for pattern in row_patterns:
            match = re.search(pattern, base, re.IGNORECASE)
            if match:
                self.features['row_num'] = match.group(1).zfill(2)
                break

        col_patterns = [
            r'COL\s*[_-]?\s*(\d+)', r'列\s*[_-]?\s*(\d+)',
            r'C\s*[_-]?\s*(\d+)', r'(\d+)\s*列'
        ]
        for pattern in col_patterns:
            match = re.search(pattern, base, re.IGNORECASE)
            if match:
                self.features['col_num'] = match.group(1).zfill(2)
                break

        if not self.features['row_num'] or not self.features['col_num']:
            num_match = re.findall(r'(\d+)', base)
            if len(num_match) >= 2:
                if not self.features['row_num']:
                    self.features['row_num'] = num_match[-2].zfill(2)
                if not self.features['col_num']:
                    self.features['col_num'] = num_match[-1].zfill(2)

        if self.features['row_num'] and self.features['col_num']:
            self.features['row_col'] = f"{self.features['row_num']}_{self.features['col_num']}"
            self.features['grid_id'] = f"R{self.features['row_num']}C{self.features['col_num']}"

    def _extract_path_info(self):
        """从路径中提取划分和日期信息"""
        path_parts = Path(self.file_path).parts

        for part in path_parts:
            if part.lower() in self.config['TARGET_SPLITS']:
                self.features['split_type'] = part.lower()
                break

        for part in path_parts:
            digits = re.sub(r'[^0-9]', '', part)
            if len(digits) >= 6:
                self.features['date_folder'] = digits[-6:]
                break

    def _calculate_parse_score(self):
        """计算解析质量分数"""
        score = 0.0
        if self.features['full_dji_code']:
            score += 0.4
        if self.features['grid_id']:
            score += 0.3
        if self.features['channel_type']:
            score += 0.2
        if self.features['date_str']:
            score += 0.1

        self.features['parse_score'] = score
        self.features['parse_success'] = score >= 0.3

    def get_all_match_keys(self):
        """获取所有可能的匹配键"""
        keys = []
        if self.features['full_dji_code'] and self.features['grid_id']:
            keys.append(f"{self.features['full_dji_code']}_{self.features['grid_id']}")
        if self.features['grid_id']:
            keys.append(self.features['grid_id'])
        if self.features['full_dji_code']:
            keys.append(self.features['full_dji_code'])
        if self.features['sequence_number']:
            keys.append(self.features['sequence_number'])
        keys.append(str(hash(self.features['filename']) % 1000000))
        return keys

# ===================== 2. 工具函数 =====================
def parse_date_folder(folder_name):
    """从文件夹名解析日期"""
    digits = re.sub(r'[^0-9]', '', folder_name)
    if len(digits) >= 6:
        return digits[-8:] if len(digits) >= 8 else digits[-6:]
    return None

def is_ignore_path(file_path, ignore_splits):
    """检查是否应该忽略该路径"""
    for ignore in ignore_splits:
        if f"{os.sep}{ignore}{os.sep}" in file_path or file_path.startswith(f"{ignore}{os.sep}"):
            return True
    return False

# ===================== 3. 核心匹配函数 =====================
def scan_and_match_samples(config=None):
    """扫描并匹配RGB和多光谱样本"""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    final_config = DEFAULT_CONFIG.copy()
    final_config.update(config)
    config = final_config

    print(f"开始扫描目录: {os.path.abspath(config['DATASET_ROOT'])}")

    channel_files = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    reverse_index = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    scanned_count = 0
    processed_count = 0
    valid_feature_count = 0

    # 第一步：扫描文件
    for root, _, files in os.walk(config['DATASET_ROOT']):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()

            if is_ignore_path(file_path, config['IGNORE_SPLITS']) or file_ext not in config['SUPPORTED_EXTS']:
                continue
            scanned_count += 1

            extractor = DJIFileFeatureExtractor(file_path, config)
            features = extractor.features

            if not features['split_type'] or not features['date_folder']:
                continue

            path_parts = Path(file_path).parts
            for part in path_parts:
                if part.upper() in [config['RGB_CHANNEL'].upper()] + [ch.upper() for ch in config['MS_CHANNELS']]:
                    features['channel_type'] = part.upper()
                    break

            if not features['channel_type']:
                continue

            match_keys = extractor.get_all_match_keys()
            main_key = match_keys[0]
            split = features['split_type']
            date = features['date_folder']
            channel = features['channel_type']

            channel_files[split][date][channel][main_key] = {
                'filename': features['filename'],
                'path': file_path,
                'features': features,
                'all_keys': match_keys
            }

            for key in match_keys:
                reverse_index[channel][split][date][key] = {
                    'filename': features['filename'],
                    'path': file_path,
                    'features': features
                }

            processed_count += 1
            if features['parse_success']:
                valid_feature_count += 1

    print(f"\n=== 扫描统计 ===")
    print(f"总有效图像文件数: {scanned_count}")
    print(f"处理文件数（train/val）: {processed_count}")
    print(f"有效特征提取数: {valid_feature_count} ({valid_feature_count/processed_count*100:.1f}%)")

    # 第二步：智能匹配
    complete_samples = []
    date_stats = defaultdict(lambda: defaultdict(int))
    required_channels = [config['RGB_CHANNEL'].upper()] + [ch.upper() for ch in config['MS_CHANNELS']]

    print(f"\n=== 匹配结果统计 ===")
    for split in channel_files:
        for date in channel_files[split]:
            rgb_channel_data = channel_files[split][date].get(config['RGB_CHANNEL'].upper(), {})
            if not rgb_channel_data:
                continue

            for rgb_key, rgb_info in rgb_channel_data.items():
                sample_channels = {config['RGB_CHANNEL'].upper(): rgb_info}
                missing_channels = []
                match_key_used = None

                for ms_channel in config['MS_CHANNELS']:
                    ms_channel_upper = ms_channel.upper()
                    ms_channel_data = channel_files[split][date].get(ms_channel_upper, {})

                    if rgb_key in ms_channel_data:
                        sample_channels[ms_channel_upper] = ms_channel_data[rgb_key]
                        match_key_used = rgb_key
                        continue

                    if config['FALLBACK_MATCH']:
                        matched = False
                        for test_key in rgb_info['all_keys']:
                            if test_key in ms_channel_data:
                                sample_channels[ms_channel_upper] = ms_channel_data[test_key]
                                match_key_used = test_key
                                matched = True
                                break
                            if test_key in reverse_index[ms_channel_upper][split][date]:
                                sample_channels[ms_channel_upper] = reverse_index[ms_channel_upper][split][date][test_key]
                                match_key_used = test_key
                                matched = True
                                break
                        if not matched:
                            missing_channels.append(ms_channel_upper)

                if not missing_channels or (config['MISSING_CHANNEL_ALLOW'] and len(missing_channels) < len(config['MS_CHANNELS'])):
                    sample = {
                        '样本ID': f"{split}_{date}_{match_key_used if match_key_used else 'unknown'}",
                        '数据集划分': split,
                        '日期': date,
                        '匹配键': match_key_used if match_key_used else 'unknown',
                        '缺失通道': ', '.join(missing_channels) if missing_channels else '无'
                    }

                    for ch in required_channels:
                        if ch in sample_channels:
                            data = sample_channels[ch]
                            sample[f'{ch}文件名'] = data['filename']
                            sample[f'{ch}路径'] = data['path']
                            sample[f'{ch}_DJI编码'] = data['features'].get('full_dji_code', '')
                            sample[f'{ch}_网格ID'] = data['features'].get('grid_id', '')
                        else:
                            sample[f'{ch}文件名'] = ''
                            sample[f'{ch}路径'] = ''
                            sample[f'{ch}_DJI编码'] = ''
                            sample[f'{ch}_网格ID'] = ''

                    complete_samples.append(sample)
                    date_stats[date][split] += 1

            print(f"  日期 {date} 匹配完成：{date_stats[date][split]} 个样本")

    total_matched = len(complete_samples)
    print(f"\n总计匹配完整样本数: {total_matched}")

    if total_matched > 0:
        print("\n各日期匹配样本数:")
        for date in sorted(date_stats.keys()):
            print(f"  日期 {date}:")
            for split in sorted(config['TARGET_SPLITS']):
                count = date_stats[date].get(split, 0)
                if count > 0:
                    print(f"    {split}集: {count} 个")
    else:
        print("\n⚠️  未匹配到完整样本！")

    return complete_samples

# ===================== 4. CSV导出函数（修复版） =====================
def export_csv(samples, output_path=None, config=None):
    """
    将匹配结果导出为CSV文件

    参数:
        samples: 样本列表
        output_path: 输出路径（可选）
        config: 配置字典

    返回:
        bool: 是否成功导出
    """
    if not samples:
        print("⚠️  无样本可导出，跳过CSV生成")
        return False

    if config is None:
        config = DEFAULT_CONFIG

    # 设置默认输出路径
    if output_path is None:
        output_path = config.get('CSV_OUTPUT_PATH', 'rgbms_samples_list.csv')

    # 确保输出路径是字符串
    if not isinstance(output_path, str):
        output_path = str(output_path)

    # 获取输出目录
    output_dir = os.path.dirname(output_path)

    # 修复：只有在有目录部分时才创建目录
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"创建输出目录: {output_dir}")
        except Exception as e:
            print(f"⚠️  创建目录失败: {e}")
            # 尝试在当前目录创建文件
            output_path = os.path.basename(output_path)
            print(f"改为在当前目录创建: {output_path}")

    required_channels = [config['RGB_CHANNEL'].upper()] + [ch.upper() for ch in config['MS_CHANNELS']]
    header = ['样本ID', '数据集划分', '日期', '匹配键', '缺失通道']

    for ch in required_channels:
        header.append(f'{ch}文件名')
        header.append(f'{ch}路径')
        header.append(f'{ch}_DJI编码')
        header.append(f'{ch}_网格ID')

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(samples)

        print(f"\n✅ 匹配结果已导出至: {os.path.abspath(output_path)}")
        print(f"📊 导出样本数: {len(samples)}")
        print(f"📄 文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")

        return True
    except Exception as e:
        print(f"❌ 导出CSV失败: {e}")
        return False

# ===================== 5. 主执行函数 =====================
def main():
    """独立运行数据匹配"""
    print("=" * 80)
    print("RGB-MS数据匹配工具")
    print("=" * 80)

    config = {
        'DATASET_ROOT': 'data-RGBMS',
        'CSV_OUTPUT_PATH': "rgbms_samples_list.csv",
        'MISSING_CHANNEL_ALLOW': False,
    }

    samples = scan_and_match_samples(config)

    if samples:
        success = export_csv(samples, config['CSV_OUTPUT_PATH'], config)
        if success:
            print(f"\n🎉 数据匹配完成！共匹配到 {len(samples)} 个样本。")
            return True
        else:
            print("\n❌ CSV导出失败！")
            return False
    else:
        print("\n❌ 未匹配到任何样本！")
        return False

# ===================== 6. 简易导出函数 =====================
def quick_export(samples, filename="rgbms_samples.csv"):
    """
    快速导出函数（简化版）

    参数:
        samples: 样本列表
        filename: 输出文件名
    """
    if not samples:
        print("没有样本可导出")
        return False

    try:
        # 自动推断字段
        if samples:
            sample_keys = samples[0].keys()
        else:
            sample_keys = []

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sample_keys)
            writer.writeheader()
            writer.writerows(samples)

        print(f"成功导出 {len(samples)} 个样本到 {filename}")
        return True
    except Exception as e:
        print(f"导出失败: {e}")
        return False

if __name__ == '__main__':
    main()