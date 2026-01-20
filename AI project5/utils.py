# utils.py
def load_label_file(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # 尝试 gbk（Windows 常见）
        with open(filepath, 'r', encoding='gbk') as f:
            lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        
        # 支持两种分隔符：逗号 或 空格
        if ',' in line:
            parts = line.split(',', maxsplit=1)  # 只按第一个逗号切
        else:
            parts = line.split(maxsplit=1)       # 按空白符切，最多切两部分
        
        if len(parts) >= 2:
            guid = parts[0].strip()
            label = parts[1].strip()
            if label in {'positive', 'neutral', 'negative'}:
                data.append((guid, label))
            else:
                print(f"⚠️ 警告：无效标签 '{label}'，跳过该行: {line}")
        else:
            print(f"⚠️ 警告：格式错误，跳过该行: {line}")
    
    print(f"✅ 成功加载 {len(data)} 条训练样本")
    return data

def split_train_val(train_file, val_ratio=0.2, random_state=42):
    data = load_label_file(train_file)
    if len(data) == 0:
        raise ValueError(f"❌ 未加载到任何有效样本！请检查文件 '{train_file}' 是否存在且格式正确。")
    
    labels = [d[1] for d in data]
    from sklearn.model_selection import train_test_split
    train, val = train_test_split(data, test_size=val_ratio, random_state=random_state, stratify=labels)
    return train, val

def load_test_guids(test_file):
    """仅加载 test_without_label.txt 中的 guid，忽略 label（应为 'null'）"""
    guids = []
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(test_file, 'r', encoding='gbk') as f:
            lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 支持逗号或空格分隔
        if ',' in line:
            guid = line.split(',')[0].strip()
        else:
            guid = line.split()[0].strip()
        if guid and guid != 'guid':  # 跳过可能的表头
            guids.append(guid)
    
    print(f"✅ 成功加载 {len(guids)} 个测试样本 guid")
    return guids