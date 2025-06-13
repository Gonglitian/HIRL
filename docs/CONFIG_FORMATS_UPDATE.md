# 配置文件数据格式更新说明

## 🔄 更新内容

已将所有配置文件中的默认数据格式从 `pickle` 更新为 `hdf5`，并添加了完整的格式选择说明。

### 📝 更新的配置文件

1. **`configs/pusht_human.yaml`**
   - 默认格式: `pickle` → `hdf5`
   - 新注释: `# 保存格式: hdf5|json|csv|npz|pickle (推荐hdf5，纯数据无类依赖)`

2. **`configs/pusht_human_mouse.yaml`**
   - 默认格式: `pickle` → `hdf5`
   - 新注释: `# 保存格式: hdf5|json|csv|npz|pickle (推荐hdf5，纯数据无类依赖)`

3. **`configs/replay.yaml`**
   - 默认文件扩展名: `.pickle` → `.h5`
   - 新注释: `# 数据文件路径 (支持格式: .h5/.json/.csv/.npz/.pkl)`

## 🚀 可用的数据格式

| 格式 | 扩展名 | 特点 | 推荐场景 |
|------|--------|------|----------|
| **HDF5** | `.h5` | 高效压缩，纯数据，无类依赖 | **默认选择，生产环境** |
| **JSON** | `.json` | 人类可读，跨语言兼容 | 调试、小规模数据 |
| **CSV** | `.csv` | 通用格式，分析友好 | 数据分析、Excel处理 |
| **NPZ** | `.npz` | NumPy原生，快速加载 | 数值计算、训练数据 |
| **Pickle** | `.pkl` | Python对象序列化 | 不推荐（有类依赖） |

## 📋 使用示例

### 在配置文件中指定格式
```yaml
data:
  save_format: "hdf5"    # 推荐
  save_format: "json"    # 调试用
  save_format: "csv"     # 分析用
  save_format: "npz"     # 训练用
```

### 通过命令行覆盖
```bash
# 使用JSON格式进行调试
python main.py data.save_format=json

# 使用CSV格式便于分析
python main.py data.save_format=csv

# 使用NPZ格式进行训练
python main.py --config-name=pusht_human data.save_format=npz
```

### 回放不同格式的文件
```bash
# 回放HDF5文件
python replay.py data_path=data/trajectories.h5

# 回放JSON文件
python replay.py data_path=data/trajectories.json

# 回放CSV文件
python replay.py data_path=data/trajectories.csv
```

## ⚠️ 重要提示

1. **向后兼容**: 仍然支持加载旧的pickle文件
2. **推荐迁移**: 建议使用转换脚本将现有pickle文件转换为HDF5格式
3. **默认选择**: 新项目默认使用HDF5格式，无需额外配置

## 🛠️ 数据转换

如果您有现有的pickle格式数据，可以使用转换脚本：

```bash
# 转换为HDF5格式（推荐）
python scripts/convert_data_format.py old_file.pkl --format hdf5

# 转换为JSON格式
python scripts/convert_data_format.py old_file.pkl --format json
```

现在您的所有配置都默认使用纯数据格式，确保数据的长期可用性和跨环境兼容性！ 