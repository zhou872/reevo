# 从typing模块导入Optional类型，用于表示可选参数（可以是None）
from typing import Optional
# 导入logging库，用于记录程序运行日志
import logging
# 导入subprocess库，用于运行外部程序（如执行Python代码）
import subprocess
# 导入numpy库，用于数值计算（数组操作等），通常简写为np
import numpy as np
# 导入os库，用于操作系统相关的功能（如文件路径操作）
import os
# 从omegaconf库导入DictConfig类型，用于表示配置字典（hydra框架使用的配置类型）
from omegaconf import DictConfig

# 从utils.utils模块导入所有函数（*表示导入所有）
from utils.utils import *
# 从utils.llm_client.base模块导入BaseClient基类，用于定义LLM客户端的接口
from utils.llm_client.base import BaseClient

from knowledge_graph import KnowledgeGraph#新加
import re
import json#新加

# 定义ReEvo类，这是整个进化算法的核心类（ReEvo = Reflective Evolution的缩写）
class ReEvo:
    # __init__是类的初始化方法（构造函数），在创建ReEvo对象时自动调用
    def __init__(
        self, 
        cfg: DictConfig,  # 配置对象，包含所有算法参数（问题设置、种群大小等）
        root_dir: str,  # 项目根目录的路径（字符串类型）
        generator_llm: BaseClient,  # 生成器LLM客户端，用于生成代码（必需参数）
        reflector_llm: Optional[BaseClient] = None,  # 反思器LLM客户端（可选，如果为None则使用generator_llm）
        
        # 支持为四个操作设置不同的LLM：
        # 短期反思（Short-term Reflection）、长期反思（Long-term Reflection）、交叉（Crossover）、变异（Mutation）
        short_reflector_llm: Optional[BaseClient] = None,  # 短期反思LLM（可选）
        long_reflector_llm: Optional[BaseClient] = None,  # 长期反思LLM（可选）
        crossover_llm: Optional[BaseClient] = None,  # 交叉操作LLM（可选）
        mutation_llm: Optional[BaseClient] = None,  # 变异操作LLM（可选）
    ) -> None:  # 返回类型为None（构造函数不返回值）
        # 保存配置对象到实例变量（self表示当前对象实例）
        self.cfg = cfg
        # 保存生成器LLM客户端
        self.generator_llm = generator_llm
        # 如果提供了reflector_llm就使用它，否则使用generator_llm（or表示如果左边为None则用右边）
        self.reflector_llm = reflector_llm or generator_llm

        # 设置短期反思LLM：如果提供了就用提供的，否则用reflector_llm
        self.short_reflector_llm = short_reflector_llm or self.reflector_llm
        # 设置长期反思LLM：如果提供了就用提供的，否则用reflector_llm
        self.long_reflector_llm = long_reflector_llm or self.reflector_llm
        # 设置交叉操作LLM：如果提供了就用提供的，否则用generator_llm
        self.crossover_llm = crossover_llm or generator_llm
        # 设置变异操作LLM：如果提供了就用提供的，否则用generator_llm
        self.mutation_llm = mutation_llm or generator_llm

        # 保存项目根目录路径
        self.root_dir = root_dir
        
        # 从配置中读取变异率（变异操作产生的个体数量比例）
        self.mutation_rate = cfg.mutation_rate
        # 初始化迭代计数器为0（记录当前是第几次迭代）
        self.iteration = 0
        # 初始化函数评估计数器为0（记录总共执行了多少次代码评估）
        self.function_evals = 0
        # 初始化精英个体为None（精英个体是当前最好的个体，会被保留下来）
        self.elitist = None
        # 初始化长期反思字符串为空（用于存储累积的反思经验）
        self.long_term_reflection_str = ""
        # 初始化全局最佳目标函数值为None（记录整个进化过程中最好的目标值）
        self.best_obj_overall = None
        # 初始化全局最佳代码为None（记录整个进化过程中最好的代码）
        self.best_code_overall = None
        # 初始化全局最佳代码路径为None（记录全局最佳代码的文件路径）
        self.best_code_path_overall = None
        
        # 调用初始化提示词的方法（加载所有需要的提示词模板）
        self.init_prompt()

        self.kg = KnowledgeGraph(similarity_threshold=0.85)#新加

        # 调用初始化种群的方法（创建初始种群，评估种子函数）
        self.init_population()


    # 初始化提示词方法：加载所有需要的提示词模板文件
    def init_prompt(self) -> None:  # 返回类型为None
        # 从配置中读取问题名称（例如：tsp_aco、cvrp_aco等）
        self.problem = self.cfg.problem.problem_name
        # 从配置中读取问题描述（问题的文字说明）
        self.problem_desc = self.cfg.problem.description
        # 从配置中读取问题规模（例如：TSP问题的城市数量）
        self.problem_size = self.cfg.problem.problem_size
        # 从配置中读取函数名称（要生成的函数的名字）
        self.func_name = self.cfg.problem.func_name
        # 从配置中读取目标函数类型（"min"表示最小化，"max"表示最大化）
        self.obj_type = self.cfg.problem.obj_type
        # 从配置中读取问题类型（"black_box"表示黑盒问题，其他表示白盒问题）
        self.problem_type = self.cfg.problem.problem_type
        
        # 打印问题相关信息到日志
        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)
        logging.info("Function name: " + self.func_name)
        
        # 设置提示词目录路径（f-string用于字符串格式化，{}内可以是变量）
        self.prompt_dir = f"{self.root_dir}/prompts"
        # 设置输出文件路径（生成的代码会保存到这个文件）
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"
        
        # 加载所有文本提示词
        # 问题特定的提示词组件
        # 如果是黑盒问题，在路径后面加上"_black_box"后缀，否则为空字符串
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        # 构造问题特定的提示词目录路径
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'
        # 读取种子函数文件（种子函数是算法的起始代码）
        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')
        # 读取函数签名文件（定义了函数的输入输出格式）
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt')
        # 读取函数描述文件（描述函数应该做什么）
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        # 检查是否存在外部知识文件
        if os.path.exists(f'{problem_prompt_path}/external_knowledge.txt'):
            # 如果存在，读取外部知识（领域相关知识）
            self.external_knowledge = file_to_string(f'{problem_prompt_path}/external_knowledge.txt')
            # 将外部知识设置为长期反思的初始内容
            self.long_term_reflection_str = self.external_knowledge
        else:
            # 如果不存在，外部知识为空字符串
            self.external_knowledge = ""
        
        
        # 通用提示词（所有问题共用的提示词模板）
        # 读取生成器的系统提示词（定义生成器的角色和行为）
        self.system_generator_prompt = file_to_string(f'{self.prompt_dir}/common/system_generator.txt')
        # 读取反思器的系统提示词（定义反思器的角色和行为）
        self.system_reflector_prompt = file_to_string(f'{self.prompt_dir}/common/system_reflector.txt')
        # 读取短期反思的用户提示词（根据问题类型选择不同的模板）
        # 如果是黑盒问题，使用黑盒版本的提示词，否则使用普通版本
        self.user_reflector_st_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_st.txt') if self.problem_type != "black_box" else file_to_string(f'{self.prompt_dir}/common/user_reflector_st_black_box.txt') # 短期反思
        # 读取长期反思的用户提示词
        self.user_reflector_lt_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_lt.txt') # 长期反思
        # 读取交叉操作的提示词
        self.crossover_prompt = file_to_string(f'{self.prompt_dir}/common/crossover.txt')
        # 读取变异操作的提示词
        self.mutation_prompt = file_to_string(f'{self.prompt_dir}/common/mutation.txt')
        # 读取生成器的用户提示词，并使用format方法填充变量（func_name、problem_desc、func_desc）
        self.user_generator_prompt = file_to_string(f'{self.prompt_dir}/common/user_generator.txt').format(
            func_name=self.func_name,  # 函数名
            problem_desc=self.problem_desc,  # 问题描述
            func_desc=self.func_desc,  # 函数描述
            )
        # 读取种子提示词，并使用format方法填充变量（seed_func、func_name）
        self.seed_prompt = file_to_string(f'{self.prompt_dir}/common/seed.txt').format(
            seed_func=self.seed_func,  # 种子函数代码
            func_name=self.func_name,  # 函数名
        )

        # 标志位：控制是否打印提示词（只在第一次迭代时打印，避免日志过多）
        self.print_crossover_prompt = True  # 第一次迭代时打印交叉提示词
        self.print_mutate_prompt = True  # 第一次迭代时打印变异提示词
        self.print_short_term_reflection_prompt = True  # 第一次迭代时打印短期反思提示词
        self.print_long_term_reflection_prompt = True  # 第一次迭代时打印长期反思提示词


    # 初始化种群方法：创建初始种群（包含种子函数和LLM生成的其他个体）
    def init_population(self) -> None:  # 返回类型为None
        # 评估种子函数，并将其设置为精英个体
        logging.info("Evaluating seed function...")
        # 从种子函数中提取代码，并将版本号从"v1"替换为"v2"（用于区分不同版本）
        code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
        # 打印种子函数代码
        logging.info("Seed function code: \n" + code)
        # 创建种子个体的字典（包含所有必要信息）
        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",  # 标准输出文件路径
            "code_path": f"problem_iter{self.iteration}_code0.py",  # 代码文件路径
            "code": code,  # 代码内容
            "response_id": 0,  # 响应ID（种子函数的ID为0）
        }
        # 保存种子个体到实例变量
        self.seed_ind = seed_ind
        # 评估种子函数（运行代码，计算目标函数值）
        self.population = self.evaluate_population([seed_ind])

        # 如果种子函数无效（无法执行或执行失败），停止程序
        if not self.seed_ind["exec_success"]:  # exec_success表示执行是否成功
            # 抛出运行时错误，提示用户检查标准输出文件
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")

        # 更新迭代信息（更新最佳个体、精英个体等）
        self.update_iter()
        
        # 生成初始种群的其他个体（通过LLM生成）
        # 构建系统提示词
        system = self.system_generator_prompt
        # 构建用户提示词：拼接生成器提示词、种子提示词和长期反思字符串
        user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str
        # 构建消息列表（符合OpenAI API格式）：系统消息和用户消息
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # 打印初始种群生成的提示词
        logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)

        # 调用LLM生成多个响应（初始种群的大小由cfg.init_pop_size决定）
        # temperature参数增加0.3是为了增加初始种群的多样性（温度越高，生成结果越随机）
        responses = self.generator_llm.multi_chat_completion([messages], self.cfg.init_pop_size, temperature = self.generator_llm.temperature + 0.3) # 增加温度以获得更多样化的初始种群
        # 将每个响应转换为个体（字典格式），使用列表推导式
        # enumerate用于同时获得索引和值
        population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(responses)]

        # 运行代码并评估种群（计算每个个体的目标函数值）
        population = self.evaluate_population(population)

        # 更新迭代
        # 将评估后的种群保存到实例变量
        self.population = population
        # 更新迭代信息
        self.update_iter()

    """
    # 将LLM响应转换为个体（字典格式）
    def response_to_individual(self, response: str, response_id: int, file_name: str=None) -> dict:
       
        #将响应转换为个体
        #Convert response to individual
       
        # 将响应写入文件（保存LLM的原始响应）
        # 如果没有提供文件名，则使用默认格式生成文件名
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        # 打开文件并写入响应内容（'w'表示写入模式，会覆盖已有文件）
        with open(file_name, 'w') as file:
            file.writelines(response + '\n')  # 写入响应并换行

        # 从响应中提取代码（LLM的响应可能包含代码块，需要提取出来）
        code = extract_code_from_generator(response)

        # 从响应中提取代码和描述
        # 构造标准输出文件路径（如果file_name为None，使用默认格式）
        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"
        
        # 创建个体字典（包含个体的所有信息）
        individual = {
            "stdout_filepath": std_out_filepath,  # 标准输出文件路径（用于保存代码执行结果）
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",  # 代码文件路径
            "code": code,  # 提取出的代码内容
            "response_id": response_id,  # 响应ID（用于标识不同的个体）
        }
        return individual  # 返回个体字典
        """
    def response_to_individual(self, response: str, response_id: int, file_name: str=None) -> dict:
        """
        Convert response to individual.
        Modified to parse JSON and extract 'algorithm_features' (Self-Tagging).
        """
        import json
        import re
        
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w', encoding="utf-8") as file:
            file.writelines(response + '\n')

        # === Phase 3: JSON Parsing for Self-Tagging ===
        features = []
        
        try:
            # 1. 独立提取 JSON 特征标签 (寻找 ```json 块)
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                features = data.get("algorithm_features", [])
            else:
                # 降级：如果大模型忘了写 markdown，寻找第一个大括号
                start = response.find('{')
                end = response.find('}') # 只找第一个结束括号，防止把后面的代码包进去
                if start != -1 and end != -1 and start < end:
                    data = json.loads(response[start:end+1])
                    features = data.get("algorithm_features", [])
        except Exception as e:
            # 强制记录解析失败
            with open("KG_GEN_ERROR.txt", "a", encoding="utf-8") as f:
                f.write(f"Iter {self.iteration} Res {response_id} | Generator JSON 解析失败: {e}\n")

        # 2. 独立提取代码 (使用 ReEvo 原生提取器，非常稳健)
        code = extract_code_from_generator(response)

        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"
        
        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "code": code,
            "response_id": response_id,
            "features": features  # <--- 新增：个体自带特征标签
        }
        return individual

    # 标记个体为无效（当代码执行失败时调用）
    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        将个体标记为无效
        Mark an individual as invalid.
        """
        # 设置执行成功标志为False（表示执行失败）
        individual["exec_success"] = False
        # 将目标函数值设置为正无穷（float("inf")），表示这是一个很差的解
        individual["obj"] = float("inf")
        # 保存错误追踪信息（traceback_msg包含错误详情）
        individual["traceback_msg"] = traceback_msg
        return individual  # 返回更新后的个体


    # 评估种群方法：通过运行代码（并行执行）并计算目标函数值来评估种群
    def evaluate_population(self, population: list[dict]) -> list[float]:
        """
        通过并行运行代码并计算目标函数值来评估种群
        Evaluate population by running code in parallel and computing objective values.
        """
        # 存储所有子进程的列表（用于并行执行代码）
        inner_runs = []
        # 运行代码进行评估（遍历种群中的每个个体）
        for response_id in range(len(population)):
            # 增加函数评估计数器（每评估一个个体就加1）
            self.function_evals += 1
            # 如果响应无效（代码为None），跳过评估
            if population[response_id]["code"] is None:
                # 标记为无效个体
                population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
                # 在子进程列表中添加None占位符
                inner_runs.append(None)
                continue  # 跳过当前循环，继续下一个个体
            
            # 打印正在运行代码的信息
            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")
            
            try:
                # 运行代码，返回子进程对象（_run_code会启动一个子进程）
                process = self._run_code(population[response_id], response_id)
                # 将子进程添加到列表中
                inner_runs.append(process)
            except Exception as e:  # 如果代码执行失败（启动子进程时出错）
                logging.info(f"Error for response_id {response_id}: {e}")
                # 标记为无效个体
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                # 在子进程列表中添加None占位符
                inner_runs.append(None)
        
        # 用目标函数值更新种群（等待所有子进程执行完成）
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None:  # 如果代码执行失败，跳过
                continue
            try:
                # 等待代码执行完成（communicate会等待进程结束）
                # timeout参数设置超时时间，超过这个时间还没完成就抛出异常
                inner_run.communicate(timeout=self.cfg.timeout)  # 等待代码执行完成
            except subprocess.TimeoutExpired as e:  # 如果执行超时
                logging.info(f"Error for response_id {response_id}: {e}")
                # 标记为无效个体
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                # 强制终止子进程（kill方法会杀死进程）
                inner_run.kill()
                continue  # 跳过当前循环，继续下一个

            # 获取当前个体
            individual = population[response_id]
            # 获取标准输出文件路径
            stdout_filepath = individual["stdout_filepath"]
            # 读取标准输出文件（'r'表示只读模式）
            with open(stdout_filepath, 'r') as f:  # 读取标准输出文件
                stdout_str = f.read()  # 读取文件全部内容
            # 过滤出错误追踪信息（如果有错误的话）
            traceback_msg = filter_traceback(stdout_str)
            
            # 再次获取当前个体（确保使用最新的引用）
            individual = population[response_id]
            # 为每个个体存储目标函数值
            if traceback_msg == '':  # 如果执行没有错误
                try:
                    # 从标准输出的倒数第二行读取目标函数值（最后一行通常是空行）
                    # split('\n')按换行符分割，[-2]取倒数第二个元素
                    # 如果目标是最小化，直接使用该值；如果是最大化，取负值（因为算法统一处理为最小化）
                    individual["obj"] = float(stdout_str.split('\n')[-2]) if self.obj_type == "min" else -float(stdout_str.split('\n')[-2])
                    # 设置执行成功标志为True
                    individual["exec_success"] = True
                except:  # 如果无法解析目标函数值（格式错误等）
                    # 标记为无效个体
                    population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid std out / objective value!")
            else:  # 否则，提供执行追踪错误反馈
                # 标记为无效个体，并包含错误追踪信息
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            # 打印目标函数值
            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return population  # 返回更新后的种群（包含目标函数值和执行状态）


    # 运行代码方法（私有方法，以下划线开头）：将代码写入文件并运行评估脚本
    def _run_code(self, individual: dict, response_id) -> subprocess.Popen:
        """
        将代码写入文件并运行评估脚本
        Write code into a file and run eval script.
        """
        # 打印调试信息（debug级别的日志）
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        
        # 将个体的代码写入输出文件（覆盖模式）
        with open(self.output_file, 'w') as file:
            file.writelines(individual["code"] + '\n')  # 写入代码并换行

        # 执行Python文件（使用评估脚本）
        # 打开标准输出文件用于重定向输出（'w'表示写入模式）
        with open(individual["stdout_filepath"], 'w') as f:
            # 根据问题类型选择评估脚本路径
            # 如果不是黑盒问题，使用eval.py；如果是黑盒问题，使用eval_black_box.py
            eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py' if self.problem_type != "black_box" else f'{self.root_dir}/problems/{self.problem}/eval_black_box.py' 
            # 使用subprocess.Popen启动子进程执行评估脚本
            # 'python'是命令，'-u'表示无缓冲输出（立即显示输出）
            # eval_file_path是评估脚本路径，后面是传递给脚本的参数：问题规模、根目录、"train"（表示训练模式）
            # stdout=f和stderr=f表示将标准输出和标准错误都重定向到文件f
            process = subprocess.Popen(['python', '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train"],
                                        stdout=f, stderr=f)

        # 阻塞直到代码开始运行（等待文件被创建，表示代码已经开始执行）
        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=response_id)
        return process  # 返回子进程对象（可以在后续等待其完成）

    
    # 更新迭代方法：在每次迭代后更新最佳个体、精英个体等信息
    def update_iter(self) -> None:
        """
        每次迭代后更新
        Update after each iteration
        """
        # 获取当前种群
        population = self.population
        # 提取所有个体的目标函数值（列表推导式）
        objs = [individual["obj"] for individual in population]
        # 找到最佳目标函数值和对应的索引
        # min(objs)返回最小值，np.argmin返回最小值的索引位置
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))
        
        # 更新全局最佳（如果当前最佳比全局最佳更好，或者还没有全局最佳）
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            # 更新全局最佳目标函数值
            self.best_obj_overall = best_obj
            # 更新全局最佳代码
            self.best_code_overall = population[best_sample_idx]["code"]
            # 更新全局最佳代码路径
            self.best_code_path_overall = population[best_sample_idx]["code_path"]
        
        # 更新精英个体（精英个体是当前迭代中最好的个体）
        # 如果还没有精英个体，或者当前最佳比精英个体更好
        if self.elitist is None or best_obj < self.elitist["obj"]:
            # 更新精英个体（保存整个个体字典的引用）
            self.elitist = population[best_sample_idx]
            # 打印精英个体的目标函数值
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")
        
        # 将代码路径转换为响应路径（用于显示超链接）
        # 将.py替换为.txt，将code替换为response
        best_path = self.best_code_path_overall.replace(".py", ".txt").replace("code", "response")
        # 打印全局最佳目标函数值和代码路径（以超链接形式显示）
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {print_hyperlink(best_path, self.best_code_path_overall)}")
        # 打印迭代完成信息
        logging.info(f"Iteration {self.iteration} finished...")
        # 打印函数评估次数
        logging.info(f"Function Evals: {self.function_evals}")
        # 迭代计数器加1（准备进入下一次迭代）
        self.iteration += 1
        
    # 基于排名的选择方法：根据排名按概率选择个体（排名越靠前，被选中的概率越高）
    def rank_select(self, population: list[dict]) -> list[dict]:
        """
        基于排名的选择，选择个体的概率与其排名成正比
        Rank-based selection, select individuals with probability proportional to their rank.
        """
        # 如果是黑盒问题，只保留执行成功且目标函数值比种子函数更好的个体
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            # 否则，只保留执行成功的个体
            population = [individual for individual in population if individual["exec_success"]]
        # 如果有效个体少于2个，无法进行选择，返回None
        if len(population) < 2:
            return None
        # 按目标函数值排序种群（从小到大，值越小越好）
        population = sorted(population, key=lambda x: x["obj"])  # lambda定义匿名函数，x["obj"]是排序的依据
        # 生成排名列表（0, 1, 2, ...）
        ranks = [i for i in range(len(population))]
        # 计算每个个体的选择概率（排名越靠前，概率越大）
        # 公式：1 / (rank + 1 + len(population))，rank越小，概率越大
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        # 归一化概率（确保所有概率之和为1）
        probs = [prob / sum(probs) for prob in probs]
        # 初始化选中的种群列表
        selected_population = []
        # 初始化尝试次数计数器
        trial = 0
        # 循环选择，直到选中足够多的个体（需要2 * pop_size个个体，因为每次选2个作为父母）
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1  # 尝试次数加1
            # 根据概率随机选择2个个体作为父母（replace=False表示不放回，不能选同一个个体两次）
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            # 如果两个父母的目标函数值不同（不是同一个个体），则添加到选中种群
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)  # extend用于将列表中的所有元素添加到另一个列表
            # 如果尝试次数超过1000次还没选够，返回None（防止无限循环）
            if trial > 1000:
                return None
        return selected_population  # 返回选中的种群
    
    
    # 随机选择方法：以相等的概率选择个体（每个个体被选中的概率相同）
    def random_select(self, population: list[dict]) -> list[dict]:
        """
        随机选择，以相等的概率选择个体
        Random selection, select individuals with equal probability.
        """
        # 初始化选中的种群列表
        selected_population = []
        # 消除无效个体（过滤掉执行失败的个体）
        if self.problem_type == "black_box":
            # 如果是黑盒问题，只保留执行成功且比种子函数更好的个体
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            # 否则，只保留执行成功的个体
            population = [individual for individual in population if individual["exec_success"]]
        # 如果有效个体少于2个，无法进行选择，返回None
        if len(population) < 2:
            return None
        # 初始化尝试次数计数器
        trial = 0
        # 循环选择，直到选中足够多的个体（需要2 * pop_size个个体）
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1  # 尝试次数加1
            # 随机选择2个个体作为父母（不指定概率，表示每个个体被选中的概率相等）
            parents = np.random.choice(population, size=2, replace=False)
            # 如果两个父母的目标函数值不同（不是同一个个体），则添加到选中种群
            # 如果两个个体的目标函数值相同，则认为它们是相同的，不需要同时选择
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)  # 将两个父母添加到选中种群
            # 如果尝试次数超过1000次还没选够，返回None（防止无限循环）
            if trial > 1000:
                return None
        return selected_population  # 返回选中的种群

    # 生成短期反思提示词方法：为交叉操作前的两个个体生成短期反思提示词
    def gen_short_term_reflection_prompt(self, ind1: dict, ind2: dict) -> tuple[list[dict], str, str]:
        """
        在交叉两个个体之前进行短期反思
        Short-term reflection before crossovering two individuals.
        """
        # 检查两个个体的目标函数值是否相同（不应该相同，因为需要区分好坏）
        if ind1["obj"] == ind2["obj"]:
            # 如果相同，打印两个个体的代码用于调试
            print(ind1["code"], ind2["code"])
            # 抛出错误（因为交叉需要区分哪个更好、哪个更差）
            raise ValueError("Two individuals to crossover have the same objective value!")
        # 确定哪个个体更好，哪个更差（目标函数值越小越好）
        if ind1["obj"] < ind2["obj"]:
            # ind1的目标值更小，所以ind1更好，ind2更差
            better_ind, worse_ind = ind1, ind2
        else:  # 在极少数情况下，两个个体可能有相同的目标函数值（但上面已经检查过了）
            # ind2的目标值更小，所以ind2更好，ind1更差
            better_ind, worse_ind = ind2, ind1

        # 过滤代码（去除注释、空行等，只保留核心代码）
        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])
        
        # 构建系统提示词（定义反思器的角色）
        system = self.system_reflector_prompt
        # 构建用户提示词，使用format方法填充变量（将占位符替换为实际值）
        user = self.user_reflector_st_prompt.format(
            func_name = self.func_name,  # 函数名
            func_desc = self.func_desc,  # 函数描述
            problem_desc = self.problem_desc,  # 问题描述
            worse_code=worse_code,  # 较差的代码
            better_code=better_code  # 较好的代码
            )
        # 构建消息列表（符合LLM API格式）
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # 如果是第一次迭代，打印反思提示词
        if self.print_short_term_reflection_prompt:
                logging.info("Short-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                # 设置为False，之后不再打印
                self.print_short_term_reflection_prompt = False
        # 返回消息列表、较差代码、较好代码（元组格式）
        return message, worse_code, better_code

    """
    # 短期反思方法：对选中的种群进行短期反思（在交叉操作之前）
    def short_term_reflection(self, population: list[dict]) -> tuple[list[list[dict]], list[str], list[str]]:
        
        #在交叉两个个体之前进行短期反思
        #Short-term reflection before crossovering two individuals.
       
        # 初始化消息列表、较差代码列表、较好代码列表
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        # 遍历种群，每次处理两个个体（range的步长为2）
        for i in range(0, len(population), 2):
            # 选择两个个体（作为父母）
            parent_1 = population[i]  # 第一个个体
            parent_2 = population[i+1]  # 第二个个体
            
            # 短期反思（生成反思提示词）
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            # 将生成的消息、较差代码、较好代码添加到对应列表
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)
        
        # 异步生成响应（批量调用LLM，提高效率）
        # multi_chat_completion可以同时处理多个对话请求
        response_lst = self.short_reflector_llm.multi_chat_completion(messages_lst)
        # 返回反思响应列表、较差代码列表、较好代码列表
        return response_lst, worse_code_lst, better_code_lst
    """

    def short_term_reflection(self, population: list[dict]) -> tuple[list[str], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        Modified to extract Knowledge Graph nodes simultaneously via JSON.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            parent_1 = population[i]
            parent_2 = population[i+1]
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)
        
        # Asynchronously generate responses
        response_lst = self.short_reflector_llm.multi_chat_completion(messages_lst)
        
        # ==========================================
        # === 核心逻辑：解析 JSON 并存入知识图谱 ===
        # ==========================================
        import json
        import re
        
        clean_reflection_texts = []
        for raw_response in response_lst:
            try:
                # 1. 终极正则提取 (完美跳过 <think> 过程)
                json_str = ""
                json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                else:
                    start = raw_response.find('{')
                    end = raw_response.rfind('}')
                    if start != -1 and end != -1:
                        json_str = raw_response[start:end+1]
                
                # 2. 解析 JSON
                data = json.loads(json_str)
                
                # 3. 获取纯文本反思
                reflection_text = data.get("reflection", "")
                if not reflection_text:
                    reflection_text = raw_response 
                clean_reflection_texts.append(reflection_text)
                
                # 4. 提取节点并存入图谱
                nodes = data.get("kg_nodes", [])
                
                # 【强制写入本地文件，绝对不会丢失】
                # 它会生成在和 response_X.txt 相同的文件夹里
                with open("KG_SUCCESS.txt", "a", encoding="utf-8") as f:
                    f.write(f"⭐⭐⭐ 成功提取 {len(nodes)} 个节点！特征: {[n.get('feature') for n in nodes]}\n")
                
                for node in nodes:
                    feature = node.get("feature")
                    logic = node.get("logic")
                    if feature and logic:
                        # 终极防爆写法：直接按位置传参！
                        self.kg.add_node(feature, logic)

            except Exception as e:
                # 【解析失败记录】
                with open("KG_ERROR.txt", "a", encoding="utf-8") as f:
                    f.write(f"🔴🔴🔴 解析失败: {e}\n")
                clean_reflection_texts.append(raw_response)

        return clean_reflection_texts, worse_code_lst, better_code_lst

    # 长期反思方法：在变异操作之前进行长期反思（整合之前的短期反思）
    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """
        在变异之前进行长期反思
        Long-term reflection before mutation.
        """
        # 构建系统提示词
        system = self.system_reflector_prompt
        # 构建用户提示词，使用format方法填充变量
        user = self.user_reflector_lt_prompt.format(
            problem_desc = self.problem_desc,  # 问题描述
            prior_reflection = self.long_term_reflection_str,  # 之前的长期反思（累积的经验）
            new_reflection = "\n".join(short_term_reflections),  # 新的短期反思（用换行符连接所有短期反思）
            )
        # 构建消息列表
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # 如果是第一次迭代，打印长期反思提示词
        if self.print_long_term_reflection_prompt:
            logging.info("Long-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            # 设置为False，之后不再打印
            self.print_long_term_reflection_prompt = False
        
        # 调用长期反思LLM生成新的长期反思（整合短期反思和之前的长期反思）
        # [messages]表示一个包含消息的列表（因为multi_chat_completion需要列表的列表）
        # [0]表示取第一个（也是唯一的）响应
        self.long_term_reflection_str = self.long_reflector_llm.multi_chat_completion([messages])[0]
        
        # 将反思结果写入文件（用于记录和调试）
        # 保存短期反思列表到文件
        file_name = f"problem_iter{self.iteration}_short_term_reflections.txt"
        with open(file_name, 'w') as file:
            file.writelines("\n".join(short_term_reflections) + '\n')  # 用换行符连接所有短期反思并写入
        
        # 保存长期反思到文件
        file_name = f"problem_iter{self.iteration}_long_term_reflection.txt"
        with open(file_name, 'w') as file:
            file.writelines(self.long_term_reflection_str + '\n')  # 写入长期反思字符串


    # 交叉方法：根据短期反思对两个个体进行交叉操作，生成新的个体
    def crossover(self, short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]) -> list[dict]:
        # 解包元组，获取短期反思内容列表、较差代码列表、较好代码列表
        reflection_content_lst, worse_code_lst, better_code_lst = short_term_reflection_tuple
        # 初始化消息列表（用于批量调用LLM）
        messages_lst = []
        # 使用zip同时遍历三个列表（对应位置的元素一起处理）
        for reflection, worse_code, better_code in zip(reflection_content_lst, worse_code_lst, better_code_lst):
            # 交叉操作
            # 构建系统提示词（定义生成器的角色）
            system = self.system_generator_prompt
            # 格式化函数签名，生成版本0和版本1的签名（用于交叉操作中定义两个父函数）
            func_signature0 = self.func_signature.format(version=0)  # 第一个父函数的签名
            func_signature1 = self.func_signature.format(version=1)  # 第二个父函数的签名
            # 构建用户提示词，使用format方法填充变量
            user = self.crossover_prompt.format(
                user_generator = self.user_generator_prompt,  # 用户生成器提示词
                func_signature0 = func_signature0,  # 第一个函数签名
                func_signature1 = func_signature1,  # 第二个函数签名
                worse_code = worse_code,  # 较差的代码（一个父函数）
                better_code = better_code,  # 较好的代码（另一个父函数）
                reflection = reflection,  # 短期反思内容（指导如何组合两个父函数）
                func_name = self.func_name,  # 函数名
            )
            # 构建消息列表（符合LLM API格式）
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            # 将消息添加到消息列表
            messages_lst.append(messages)
            
            # 如果是第一次迭代，打印交叉提示词
            if self.print_crossover_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                # 设置为False，之后不再打印
                self.print_crossover_prompt = False
        
        # 异步生成响应（批量调用LLM，提高效率）
        response_lst = self.crossover_llm.multi_chat_completion(messages_lst)
        # 将每个响应转换为个体（字典格式），使用列表推导式
        crossed_population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(response_lst)]

        # 断言：检查交叉后的种群大小是否等于配置的种群大小（如果不等会抛出异常）
        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population  # 返回交叉后的种群

    """
    # 变异方法：基于精英个体进行变异，生成新的个体
    def mutate(self) -> list[dict]:
        #基于精英的变异。我们只对最好的个体进行变异以生成n_pop个新个体。
        #Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals.
        # 构建系统提示词
        system = self.system_generator_prompt
        # 格式化函数签名，生成版本1的签名（用于变异操作）
        func_signature1 = self.func_signature.format(version=1) 
        # 构建用户提示词，使用format方法填充变量
        user = self.mutation_prompt.format(
            user_generator = self.user_generator_prompt,  # 用户生成器提示词
            #reflection = self.long_term_reflection_str + self.external_knowledge,  # 长期反思和外部知识（指导如何改进代码）
            reflection =  self.external_knowledge,
            func_signature1 = func_signature1,  # 函数签名
            elitist_code = filter_code(self.elitist["code"]),  # 精英个体的代码（最好的代码，作为变异的基础）
            func_name = self.func_name,  # 函数名
        )
        # 构建消息列表
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # 如果是第一次迭代，打印变异提示词
        if self.print_mutate_prompt:
            logging.info("Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            # 设置为False，之后不再打印
            self.print_mutate_prompt = False
        # 调用变异LLM生成多个响应
        # int(self.cfg.pop_size * self.mutation_rate)计算要生成的个体数量（种群大小乘以变异率）
        responses = self.mutation_llm.multi_chat_completion([messages], int(self.cfg.pop_size * self.mutation_rate))
        # 将每个响应转换为个体（字典格式），使用列表推导式
        population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(responses)]
        return population  # 返回变异后的种群
    """

    def mutate(self) -> list[dict]:
        """Elitist-based mutation. Modified to inject Knowledge Graph suggestions."""
        system = self.system_generator_prompt
        func_signature1 = self.func_signature.format(version=1) 
        
        # ==========================================
        # === Phase 3: 检索知识图谱 (Retrieval) ===
        # ==========================================
        elitist_features = self.elitist.get("features", [])
        
        # ---> 【新增防御机制：给种子代码赋予默认标签】 <---
        if not elitist_features:
            elitist_features = ["Greedy Strategy", "Constructive Heuristic", "Look-ahead Mechanism"]
            logging.info("检测到种子代码，已赋予默认检索特征。")
        
        # ---> 【强制写入检索前的状态】 <---
        with open("KG_RETRIEVE_LOG.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- Iteration {self.iteration} ---\n")
            f.write(f"当前图谱节点总数: {len(self.kg.graph.nodes)}\n")
            f.write(f"实际用于检索的特征: {elitist_features}\n")
            
        suggestions = self.kg.retrieve_suggestions(elitist_features, top_k=3)
        
        # ---> 【强制写入检索后的结果】 <---
        with open("KG_RETRIEVE_LOG.txt", "a", encoding="utf-8") as f:
            f.write(f"匹配到 {len(suggestions)} 条建议！\n")
            if suggestions:
                f.write(f"建议列表: {[s['feature'] for s in suggestions]}\n")
        
        if suggestions:
            suggestion_texts = ["\n[Long-term Suggestions from Knowledge Graph]"]
            suggestion_texts.append("Here are some successful algorithmic strategies from past iterations that might help improve this code:")
            for idx, sug in enumerate(suggestions):
                suggestion_texts.append(f"{idx+1}. Feature: {sug['feature']}\n   Implementation Logic: {sug['logic']}")
            reflection_str = "\n".join(suggestion_texts) + "\n\n" + self.external_knowledge
        else:
            reflection_str = self.external_knowledge
        # ==========================================

        user = self.mutation_prompt.format(
            user_generator = self.user_generator_prompt,
            reflection = reflection_str, 
            func_signature1 = func_signature1,
            elitist_code = filter_code(self.elitist["code"]),
            func_name = self.func_name,
        )
        
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        if self.print_mutate_prompt:
            logging.info("Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_mutate_prompt = False
            
        responses = self.mutation_llm.multi_chat_completion([messages], int(self.cfg.pop_size * self.mutation_rate))
        population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(responses)]
        
        # === Phase 4 伏笔：保存突变时的上下文 ===
        used_sugs_info = [{"id": sug["id"], "feature": sug["feature"], "logic": sug["logic"]} for sug in suggestions] if suggestions else []
        for ind in population:
            ind["used_suggestions"] = used_sugs_info
            ind["parent_obj"] = self.elitist["obj"] 
            ind["parent_code"] = self.elitist["code"] 

        return population

    # 归因分析方法：根据子代与父代的 objective value，反向更新图谱权重  新增
    def assign_kg_credit(self, population: list[dict]):
        """
        Phase 4.1: 构建归因分析器，调用 LLM 进行打分
        """
        import json
        import re
        import os
        import logging
        
        # 获取 reevo.py 所在的真实目录 (项目根目录)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(base_dir, "prompts", "common", "attribution.txt")
        
        if not os.path.exists(prompt_path):
            logging.error(f"找不到归因 Prompt 文件: {prompt_path}")
            return
            
        with open(prompt_path, "r", encoding="utf-8") as f:
            attribution_prompt_template = f.read()
            
        messages_lst = []
        valid_inds = []
        
        # 准备批量调用 LLM 的数据
        for ind in population:
            used_sugs = ind.get("used_suggestions", [])
            parent_obj = ind.get("parent_obj")
            current_obj = ind.get("obj")
            
            if not used_sugs or parent_obj is None or current_obj is None:
                continue
                
            sugs_text = ""
            for sug in used_sugs:
                sugs_text += f"- ID: {sug['id']}\n  Feature: {sug['feature']}\n  Logic: {sug['logic']}\n"
                
            user_msg = attribution_prompt_template.format(
                parent_score=parent_obj,
                parent_code=ind.get("parent_code", "N/A"),
                child_score=current_obj,
                child_code=ind.get("code", "N/A"),
                suggestions_text=sugs_text
            )
            
            messages_lst.append([{"role": "user", "content": user_msg}])
            valid_inds.append(ind)
            
        if not messages_lst:
            return
            
        # 批量调用 LLM
        logging.info(f"正在启动 LLM 归因分析，共评估 {len(messages_lst)} 个子代...")
        responses = self.mutation_llm.multi_chat_completion(messages_lst)
        
        # 解析 JSON 并更新权重
        for raw_response in responses:
            try:
                json_str = ""
                json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    start = raw_response.find('{')
                    end = raw_response.rfind('}')
                    if start != -1 and end != -1:
                        json_str = raw_response[start:end+1]
                        
                data = json.loads(json_str)
                scores_dict = data.get("scores", {})
                
                # 把 LLM 打的分数喂给图谱的指数映射函数
                if scores_dict:
                    self.kg.update_weights(scores_dict)
                    
            except Exception as e:
                logging.warning(f"归因 JSON 解析失败: {e}")

    # 进化方法：整个进化算法的核心循环，执行选择、交叉、变异等操作
    def evolve(self):
        # 主循环：当函数评估次数小于最大函数评估次数时继续迭代
        while self.function_evals < self.cfg.max_fe:  # max_fe是最大函数评估次数
            # 如果所有个体都无效，停止程序
            # all()函数检查列表中的所有元素是否都为True
            # [not individual["exec_success"] for individual in self.population]生成一个布尔列表，表示每个个体是否执行失败
            if all([not individual["exec_success"] for individual in self.population]):
                # 抛出运行时错误，提示用户检查标准输出文件
                raise RuntimeError(f"All individuals are invalid. Please check the stdout files in {os.getcwd()}.")
            # 选择阶段：从种群中选择个体作为父母
            # 如果精英个体不在当前种群中，将其添加到选择池中（确保精英个体可以参与选择）
            population_to_select = self.population if (self.elitist is None or self.elitist in self.population) else [self.elitist] + self.population  # 将精英个体添加到种群中进行选择
            # 使用随机选择方法选择个体（也可以使用rank_select）
            selected_population = self.random_select(population_to_select)
            # 如果选择失败（返回None），抛出错误
            if selected_population is None:
                raise RuntimeError("Selection failed. Please check the population.")
            # 短期反思阶段：对选中的个体进行短期反思（分析哪些代码更好，为什么更好）
            # short_term_reflection返回一个元组：(response_lst, worse_code_lst, better_code_lst)
            short_term_reflection_tuple = self.short_term_reflection(selected_population)  # (response_lst, worse_code_lst, better_code_lst)
            # 交叉阶段：根据短期反思的结果，对两个个体进行交叉操作，生成新的个体
            crossed_population = self.crossover(short_term_reflection_tuple)
            # 评估阶段：运行交叉后的个体代码，计算目标函数值
            self.population = self.evaluate_population(crossed_population)
            # 更新阶段：更新最佳个体、精英个体等信息
            self.update_iter()
            # 长期反思阶段：整合所有的短期反思，生成长期反思（累积经验）
            # short_term_reflection_tuple[0]是response_lst，包含所有短期反思的响应
            #self.long_term_reflection([response for response in short_term_reflection_tuple[0]])
            """
            # 变异阶段：基于精英个体和长期反思进行变异，生成新的个体
            mutated_population = self.mutate()
            # 评估阶段：运行变异后的个体代码，计算目标函数值
            # extend方法将评估后的变异种群添加到当前种群中（合并两个列表）
            self.population.extend(self.evaluate_population(mutated_population))
            # 更新阶段：再次更新最佳个体、精英个体等信息
            self.update_iter()
            """
            # 变异阶段：基于精英个体和长期反思进行变异，生成新的个体
            mutated_population = self.mutate()
            # 评估阶段：运行变异后的个体代码，赋予它们目标函数数值 (obj 分数)
            evaluated_mutated_population = self.evaluate_population(mutated_population)
            # ---> 【新增：调用 LLM 归因与权重更新】 <---
            # 专门拿着带分数的变异子代，去给它们使用的图谱建议打分
            self.assign_kg_credit(evaluated_mutated_population)
            # 归因完成后，将这批子代合并到当前总种群中
            self.population.extend(evaluated_mutated_population)
            # 更新阶段：再次更新最佳个体、精英个体等信息
            self.update_iter()
            # 当达到最大函数评估次数后，返回全局最佳代码和代码路径
        return self.best_code_overall, self.best_code_path_overall
    
