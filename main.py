# 导入hydra库，用于管理配置文件的框架
import hydra
# 导入logging库，用于记录程序运行过程中的日志信息
import logging 
# 导入os库，用于操作文件和目录
import os
# 从pathlib库导入Path类，用于处理文件路径（比os.path更现代化）
from pathlib import Path
# 导入subprocess库，用于运行外部程序或命令
import subprocess
# 从utils.utils模块导入两个函数：init_client用于初始化客户端，print_hyperlink用于打印超链接
from utils.utils import init_client, print_hyperlink


# 获取当前工作目录的绝对路径，保存到ROOT_DIR变量中（项目根目录）
ROOT_DIR = os.getcwd()
# 配置日志系统，设置日志级别为INFO（会显示信息级别的日志）
logging.basicConfig(level=logging.INFO)

# 这是hydra装饰器，用于自动加载配置文件
# version_base=None表示不使用版本控制
# config_path="cfg"表示配置文件在cfg目录下
# config_name="config"表示主配置文件名为config.yaml
@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    # 获取当前工作目录（注意：hydra可能会改变工作目录到输出文件夹）
    workspace_dir = Path.cwd()
    # 打印当前工作目录（以超链接形式显示，方便点击）
    logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
    # 打印项目根目录（以超链接形式显示）
    logging.info(f"Project Root: {print_hyperlink(ROOT_DIR)}")
    # 打印使用的LLM（大语言模型）信息，优先使用cfg.model，如果没有则使用cfg.llm_client.model
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    # 打印使用的算法名称
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    # 根据配置初始化LLM客户端（用于与AI模型通信）
    client = init_client(cfg)
    # 以下是可选的客户端，用于ReEvo算法的不同操作（如果有配置的话就创建，否则为None）
    # 长反思LLM客户端：用于长期反思操作
    long_ref_llm = hydra.utils.instantiate(cfg.llm_long_ref) if cfg.get("llm_long_ref") else None
    # 短反思LLM客户端：用于短期反思操作
    short_ref_llm = hydra.utils.instantiate(cfg.llm_short_ref) if cfg.get("llm_short_ref") else None
    # 交叉操作LLM客户端：用于交叉操作（类似遗传算法的交叉）
    crossover_llm = hydra.utils.instantiate(cfg.llm_crossover) if cfg.get("llm_crossover") else None
    # 变异操作LLM客户端：用于变异操作（类似遗传算法的变异）
    mutation_llm = hydra.utils.instantiate(cfg.llm_mutation) if cfg.get("llm_mutation") else None
    
    # 根据配置的算法类型，动态导入对应的算法类
    if cfg.algorithm == "reevo":
        # 如果是reevo算法，从reevo模块导入ReEvo类，并给它起个别名LHH
        from reevo import ReEvo as LHH
    elif cfg.algorithm == "ael":
        # 如果是ael算法，从baselines.ael.ga模块导入AEL类，别名为LHH
        from baselines.ael.ga import AEL as LHH
    elif cfg.algorithm == "eoh":
        # 如果是eoh算法，从baselines.eoh模块导入EoH类，别名为LHH
        from baselines.eoh import EoH as LHH
    else:
        # 如果算法类型不在上述三种中，抛出"未实现"错误
        raise NotImplementedError

    # 创建算法实例（根据算法类型使用不同的参数）
    if cfg.algorithm != "reevo":
        # 如果不是reevo算法，使用简单的参数创建实例（传入配置、根目录、客户端）
        lhh = LHH(cfg, ROOT_DIR, client)
    else:
        # 如果是reevo算法，需要传入额外的LLM客户端参数（长反思、短反思、交叉、变异）
        lhh = LHH(cfg, ROOT_DIR, client, long_reflector_llm=long_ref_llm, short_reflector_llm=short_ref_llm, 
                  crossover_llm=crossover_llm, mutation_llm=mutation_llm)
        
    # 运行进化算法，返回最佳代码和最佳代码的文件路径（evolve是进化算法的核心方法）
    best_code_overall, best_code_path_overall = lhh.evolve()
    # 打印找到的最佳代码内容
    logging.info(f"Best Code Overall: {best_code_overall}")
    # 将代码文件路径转换为响应文件路径（.py改为.txt，code改为response）
    best_path = best_code_path_overall.replace(".py", ".txt").replace("code", "response")
    # 打印最佳代码的路径（以超链接形式显示）
    logging.info(f"Best Code Path Overall: {print_hyperlink(best_path, best_code_path_overall)}")
    
    # 将最佳代码保存到gpt.py文件中（用于后续验证）
    # 'w'表示写入模式，如果文件存在会覆盖，不存在会创建
    with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py", 'w') as file:
        # 将最佳代码写入文件，并在末尾添加换行符
        file.writelines(best_code_overall + '\n')
    # 构造评估脚本的路径（用于验证代码的正确性）
    test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
    # 定义验证结果输出文件名
    test_script_stdout = "best_code_overall_val_stdout.txt"
    # 打印正在运行验证脚本的信息
    logging.info(f"Running validation script...: {print_hyperlink(test_script)}")
    # 运行验证脚本，并将输出重定向到文件中
    # subprocess.run用于执行外部命令，这里执行python脚本，参数为：脚本路径、-1、根目录、val
    with open(test_script_stdout, 'w') as stdout:
        subprocess.run(["python", test_script, "-1", ROOT_DIR, "val"], stdout=stdout)
    # 打印验证脚本执行完成的信息，并显示结果文件路径
    logging.info(f"Validation script finished. Results are saved in {print_hyperlink(test_script_stdout)}.")
    
    # 读取验证结果文件并打印到日志中
    # 'r'表示只读模式
    with open(test_script_stdout, 'r') as file:
        # 逐行读取文件内容
        for line in file.readlines():
            # 打印每一行（strip()用于去除行首行尾的空白字符）
            logging.info(line.strip())

# 这是Python的标准写法，表示只有直接运行这个脚本时才执行main函数
# 如果这个文件被其他文件导入，这里的代码不会执行
if __name__ == "__main__":
    # 调用主函数（hydra会自动注入配置参数）
    main()