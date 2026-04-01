from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators

# 定义一个数据集配置
cfg = MazeDatasetConfig(
    name="my_first_maze",      # 自定义名称
    grid_n=5,                  # 迷宫网格大小 (5x5)
    n_mazes=10,                # 生成迷宫数量
    maze_ctor=LatticeMazeGenerators.gen_dfs,  # 使用深度优先搜索算法生成
)

# 生成数据集（会自动缓存到本地，下次加载相同配置时会直接读取）
dataset = MazeDataset.from_config(cfg)

# 查看第一个迷宫并转为ASCII艺术图
print(dataset[0].as_ascii())
