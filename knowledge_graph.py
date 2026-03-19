import networkx as nx  # 导入 NetworkX，用于构建有向图结构来表示知识图谱
import numpy as np  # 导入 NumPy，用于处理向量和矩阵运算
from sentence_transformers import SentenceTransformer  # 句向量编码模型，用于把文本转为向量
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度，用于度量向量之间的相似性
import logging  # 日志模块，用于记录运行信息
import uuid  # 用于生成唯一的节点 ID


class KnowledgeGraph:
    """
    一个基于向量相似度的简单知识图谱：
    - 节点：表示某个“算法特征 / 能力点”
    - 节点属性：自然语言描述、处理逻辑、可选代码片段、权重、命中次数等
    - 通过句向量和余弦相似度自动合并“相似节点”，避免重复
    """

    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', similarity_threshold=0.80):
        """
        初始化知识图谱对象。

        :param embedding_model_name: SentenceTransformer 使用的预训练模型名称
        :param similarity_threshold: 节点相似度阈值，大于该值则认为是相同或非常相似的概念
        """
        logging.info(f"Initializing Knowledge Graph with model: {embedding_model_name}")

        # 使用有向图来表示知识图谱；此处目前只用到节点，后续可扩展边关系
        self.graph = nx.DiGraph()

        # 创建文本编码器，将自然语言特征编码为向量
        self.encoder = SentenceTransformer(embedding_model_name, device='cpu')

        # 设定相似度阈值，用于判断是否需要合并节点
        self.similarity_threshold = similarity_threshold

        # 用于缓存每个节点的向量表示，键为节点 ID，值为对应的 embedding 向量
        self.node_embeddings = {}

    def _get_embedding(self, text):
        """
        将输入文本编码为向量表示。

        :param text: 任意一段文字描述
        :return: 对应的向量（numpy 数组或类似结构）
        """
        return self.encoder.encode(text)

    def add_node(self, algorithm_feature, process_logic, code_snippet=None):
        """
        向知识图谱中新增一个“算法能力 / 特征”节点。
        - 如果与已有节点的相似度超过设定阈值，则不新建节点，而是合并到已有节点（只增加命中次数）。
        - 如果没有足够相似的节点，则创建新节点并保存其向量。

        :param algorithm_feature: 对该算法能力 / 功能点的自然语言描述
        :param process_logic: 该能力对应的处理逻辑说明（可以是文字描述）
        :param code_snippet: 可选，对应的示例代码或实现片段
        :return: 最终使用的节点 ID（可能是新建的，也可能是被合并到的旧节点）
        """
        # 1. 先根据 feature 文本计算其向量表示
        embedding = self._get_embedding(algorithm_feature)

        # 2. 在现有节点向量中查找最相近的节点及其相似度
        most_similar_node_id, max_sim = self._find_most_similar_node(embedding)

        # 3. 如果找到相似度超过阈值的节点，则合并到该节点，而不是创建新节点
        if most_similar_node_id and max_sim > self.similarity_threshold:
            logging.info(f"Found similar node {most_similar_node_id} (sim={max_sim:.4f}). Merging...")

            # 取出该节点当前存储的数据字典
            current_data = self.graph.nodes[most_similar_node_id]

            # 命中次数 +1，用于后续根据“使用频率 / 重要度”调整权重等
            current_data['hit_count'] = current_data.get('hit_count', 0) + 1

            # 返回合并到的节点 ID（即复用旧节点）
            return most_similar_node_id
        else:
            # 4. 否则创建一个全新的节点

            # 使用 UUID 生成一个相对短的唯一 ID（只取前 8 位便于阅读）
            node_id = str(uuid.uuid4())[:8]

            # 在图中新增这个节点，并附带各种属性
            self.graph.add_node(
                node_id,                 # 节点唯一 ID
                feature=algorithm_feature,  # 功能 / 特征的自然语言描述
                logic=process_logic,        # 处理逻辑说明
                code=code_snippet,          # 可选的代码片段（可以为 None）
                weight=1.0,                 # 初始权重（后续可以根据命中次数或其他因素调整）
                hit_count=1                 # 初次添加时命中次数为 1
            )

            # 将该节点的向量缓存到字典中，便于后续相似度检索
            self.node_embeddings[node_id] = embedding

            # 返回新建节点的 ID
            return node_id

    def _find_most_similar_node(self, query_embedding):
        """
        给定一个查询向量，在当前已有的节点向量中找出相似度最高的节点。

        :param query_embedding: 查询文本的向量表示
        :return: (最相似节点的 ID, 对应的最大相似度)。如果图为空，则返回 (None, 0.0)
        """
        # 如果当前还没有任何节点向量，则直接返回默认值
        if not self.node_embeddings:
            return None, 0.0

        # 取出所有节点 ID 列表
        ids = list(self.node_embeddings.keys())

        # 将所有节点的向量堆叠成一个二维矩阵，形状约为 [节点数, 向量维度]
        vectors = np.array([self.node_embeddings[nid] for nid in ids])

        # 计算查询向量与所有节点向量之间的余弦相似度
        # 这里传入 [query_embedding] 形成二维数组，返回第一行即与每个节点的相似度
        sim_matrix = cosine_similarity([query_embedding], vectors)[0]

        # 找出相似度最大的下标
        best_idx = np.argmax(sim_matrix)

        # 返回最相似节点的 ID 以及对应的相似度值
        return ids[best_idx], sim_matrix[best_idx]

    def retrieve_suggestions(self, query_features, top_k=3):
        """
        Task 3.2: 检索相关的建议节点
        :param query_features: list of strings (精英代码自带的特征标签列表)
        """
        if not self.graph.nodes or not query_features:
            return []

        # 确保输入是列表
        if isinstance(query_features, str):
            query_features = [query_features]

        # 提取图谱中所有节点的向量
        ids = list(self.node_embeddings.keys())
        vectors = np.array([self.node_embeddings[nid] for nid in ids])
        
        # 1. 计算 Query 特征的向量
        query_vecs = self.encoder.encode(query_features)
        
        # 2. 计算相似度矩阵
        # query_vecs 形状 [Q, D], vectors 形状 [N, D] -> sim_matrix 形状 [Q, N]
        sim_matrix = cosine_similarity(query_vecs, vectors)
        
        # 取每个节点在所有 Query 特征下的最大相似度（只要满足其中一个特征就召回）
        max_sim_scores = np.max(sim_matrix, axis=0)
        
        # 3. 结合图谱权重进行打分 (Weighted Scoring)
        final_scores = []
        for i, nid in enumerate(ids):
            # Phase 4 之后，有用的节点这里的 weight 会大于 1
            node_weight = self.graph.nodes[nid].get('weight', 1.0)
            score = max_sim_scores[i] * node_weight
            final_scores.append((nid, score))
            
        # 4. 排序并取 Top-K
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for nid, score in final_scores[:top_k]:
            node_data = self.graph.nodes[nid]
            results.append({
                "id": nid,
                "feature": node_data['feature'],
                "logic": node_data['logic'],
                "score": float(score)
            })
            
        # 打印一下检索结果，方便我们在终端观察
        if results:
            print(f"\n🔍 [KG 检索] 为特征 {query_features} 匹配到 {len(results)} 条建议！")
            
        return results
    
    def update_weights(self, scores_dict: dict):
        """
        Phase 4.2: 使用指数映射公式更新图谱权重
        :param scores_dict: dict, 格式为 {node_id: score}，score 由 LLM 给出，范围 [-10, 10]
        """
        for nid, score in scores_dict.items():
            if self.graph.has_node(nid):
                current_weight = self.graph.nodes[nid].get('weight', 1.0)
                
                # 指数映射公式: W_new = W_old * 2^(Score/10)
                multiplier = 2 ** (score / 10.0)
                new_weight = current_weight * multiplier
                
                # ---> 【新增：权重保底机制，防止有用节点被彻底抹杀】 <---
                new_weight = max(0.1, new_weight) 
                
                self.graph.nodes[nid]['weight'] = new_weight
                
                # 记录打分日志
                feature = self.graph.nodes[nid]['feature']
                with open("KG_WEIGHT_LOG.txt", "a", encoding="utf-8") as f:
                    f.write(f"打分: {score:3d} | 乘数: {multiplier:.2f} | 节点: [{feature[:20]}...] | 权重变动: {current_weight:.2f} -> {new_weight:.2f}\n")