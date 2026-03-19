# 从os模块导入path函数，用于处理文件路径
from os import path
# 从aco模块导入ACO类（蚁群优化算法类）
from aco import ACO
# 导入sys模块，用于访问命令行参数和系统相关的功能
import sys
# 导入numpy库，用于数值计算（数组操作等），通常简写为np
import numpy as np
# 从scipy.spatial模块导入distance_matrix函数，用于计算距离矩阵（城市之间的距离）
from scipy.spatial import distance_matrix
# 导入logging模块，用于记录日志信息
import logging
# 再次导入sys模块（重复导入，可以忽略，可能是代码历史遗留）
import sys
# 将项目根目录添加到Python的模块搜索路径中（"../../../"表示向上三级目录）
# 这样可以导入项目根目录下的模块
sys.path.insert(0, "../../../")

# 导入gpt模块（这个模块包含LLM生成的启发式函数代码）
import gpt
# 从utils.utils模块导入get_heuristic_name函数，用于获取启发式函数的名称
from utils.utils import get_heuristic_name


# 定义可能的函数名称列表（启发式函数可能有不同的版本）
# 这些名称对应gpt模块中可能存在的函数名
possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

# 从gpt模块中获取实际存在的启发式函数名称（会在possible_func_names列表中查找）
heuristic_name = get_heuristic_name(gpt, possible_func_names)
# 使用getattr函数从gpt模块中获取启发式函数对象（根据函数名获取函数本身）
# getattr(对象, 属性名)相当于 对象.属性名
heuristics = getattr(gpt, heuristic_name)

# 定义ACO算法的迭代次数为100次（算法会运行100轮）
N_ITERATIONS = 100
# 定义ACO算法的蚂蚁数量为30只（每轮有30只蚂蚁进行路径搜索）
N_ANTS = 30


# 定义solve函数：求解TSP问题的函数
def solve(node_pos):
    # node_pos参数：城市节点的位置坐标（二维数组，每行是一个城市的坐标）
    
    # 计算距离矩阵（dist_mat）：所有城市之间的距离
    # distance_matrix(node_pos, node_pos)计算node_pos中每个点与每个点之间的距离
    # 结果是一个方阵，dist_mat[i][j]表示城市i到城市j的距离
    dist_mat = distance_matrix(node_pos, node_pos)
    # 将距离矩阵对角线上的值设置为1（对角线表示城市到自己的距离，设为1避免除零错误）
    # np.diag_indices_from获取矩阵对角线的索引位置
    dist_mat[np.diag_indices_from(dist_mat)] = 1  # 将对角线设置为1（避免除零错误）
    
    # 使用启发式函数计算启发式信息矩阵（heu）
    # dist_mat.copy()创建距离矩阵的副本，避免修改原始矩阵
    # heuristics是LLM生成的启发式函数，根据距离矩阵计算启发式值
    # + 1e-9 是为了避免启发式值为0（1e-9是一个非常小的正数）
    heu = heuristics(dist_mat.copy()) + 1e-9
    # 确保启发式矩阵中的所有值都不小于1e-9（将小于1e-9的值都设为1e-9）
    # 这样可以避免除零错误，因为启发式值会在ACO算法中作为分母使用
    heu[heu < 1e-9] = 1e-9
    
    # 创建ACO算法实例
    # ACO(距离矩阵, 启发式矩阵, 蚂蚁数量)
    aco = ACO(dist_mat, heu, n_ants=N_ANTS)
    # 运行ACO算法，迭代N_ITERATIONS次，返回最优目标函数值（最短路径长度）
    obj = aco.run(N_ITERATIONS)
    # 返回目标函数值（最优解的路径长度）
    return obj

# 这是Python的标准写法，表示只有直接运行这个脚本时才执行下面的代码
# 如果这个文件被其他文件导入，这里的代码不会执行
if __name__ == "__main__":
    # 打印运行提示信息
    print("[*] Running ...")

    # 从命令行参数获取问题规模（城市数量），sys.argv[1]是第一个命令行参数
    # int()将其转换为整数类型
    problem_size = int(sys.argv[1])
    # 从命令行参数获取项目根目录路径，sys.argv[2]是第二个命令行参数
    root_dir = sys.argv[2]
    # 从命令行参数获取模式（训练或验证），sys.argv[3]是第三个命令行参数
    mood = sys.argv[3]
    # 断言：确保mood只能是'train'（训练）或'val'（验证）之一
    # 如果不是这两个值之一，程序会抛出AssertionError异常
    assert mood in ['train', 'val']

    # 构建数据集的基础路径
    # path.dirname(__file__)获取当前文件所在的目录路径
    # path.join将路径片段连接起来，得到"当前目录/dataset"
    basepath = path.join(path.dirname(__file__), "dataset")
    # 检查训练数据集文件是否存在（如果不存在，需要生成）
    # path.isfile检查文件是否存在
    if not path.isfile(path.join(basepath, "train50_dataset.npy")):
        # 如果文件不存在，从gen_inst模块导入generate_datasets函数
        from gen_inst import generate_datasets
        # 调用函数生成数据集
        generate_datasets()
    
    # 如果是训练模式
    if mood == 'train':
        # 构建训练数据集的完整路径
        # f"{mood}{problem_size}_dataset.npy" 例如："train50_dataset.npy"
        dataset_path = path.join(basepath, f"{mood}{problem_size}_dataset.npy")
        # 使用numpy加载数据集文件（.npy是numpy的二进制文件格式）
        node_positions = np.load(dataset_path)
        # 获取数据集中实例的数量（第一维的大小，即有多少个TSP问题实例）
        n_instances = node_positions.shape[0]
        # 打印数据集加载信息
        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
        
        # 初始化目标函数值列表（用于存储每个实例的求解结果）
        objs = []
        # 遍历数据集中的每个实例
        # enumerate同时返回索引i和对应的节点位置node_pos
        for i, node_pos in enumerate(node_positions):
            # 调用solve函数求解当前实例，得到目标函数值（最短路径长度）
            obj = solve(node_pos)
            # 打印当前实例的求解结果
            print(f"[*] Instance {i}: {obj}")
            # 将结果添加到列表中
            objs.append(obj)
        
        # 打印平均值提示
        print("[*] Average:")
        # 计算并打印所有实例的平均目标函数值（np.mean计算平均值）
        print(np.mean(objs))
    
    # 否则（验证模式）
    else:
        # 遍历不同的问题规模（20、50、100个城市）
        for problem_size in [20, 50, 100]:
            # 构建验证数据集的完整路径
            # 例如："val20_dataset.npy"、"val50_dataset.npy"、"val100_dataset.npy"
            dataset_path = path.join(basepath, f"{mood}{problem_size}_dataset.npy")
            # 加载验证数据集
            node_positions = np.load(dataset_path)
            # 记录日志信息（使用logging.info记录到日志文件）
            logging.info(f"[*] Evaluating {dataset_path}")
            # 获取数据集中实例的数量
            n_instances = node_positions.shape[0]
            # 初始化目标函数值列表
            objs = []
            # 遍历数据集中的每个实例
            for i, node_pos in enumerate(node_positions):
                # 调用solve函数求解当前实例
                obj = solve(node_pos)
                # 将结果添加到列表中（.item()将numpy标量转换为Python标量）
                objs.append(obj.item())
            # 打印当前问题规模的平均目标函数值
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")