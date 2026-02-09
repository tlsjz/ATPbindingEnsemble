import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef
import pickle
import os
import random
from collections import deque
import pandas as pd
import warnings
import time  # 新增：用于计时

warnings.filterwarnings('ignore')


# -------------------------- 1. 配置参数 --------------------------
class Config:
    # 数据参数
    feature_dim = 1280  # ESM-2特征维度
    num_classes = 2  # 二分类（结合/非结合）
    batch_size = 128
    test_size = 0  # 若已分开存储训练/测试集，可设为0
    random_seed = 42
    dtype = tf.int64  # 统一标签/动作数据类型（解决类型不匹配）

    # 数据路径（请修改为你的训练/测试数据文件夹路径）
    train_data_dir = "D:\\atpbinding\\atp227\\inputfile"  # 训练集分片文件夹
    test_data_dir = "D:\\atpbinding\\atp17\\inputfile"  # 测试集分片文件夹（若已分开）
    # data_dir = None  # 统一数据文件夹（train_data_dir和test_data_dir为空时生效）

    # 强化学习参数
    gamma = 0.95  # 折扣因子
    epsilon = 1.0  # 探索率
    epsilon_min = 0.01  # 最小探索率
    epsilon_decay = 0.995  # 探索率衰减
    learning_rate = 1e-4
    memory_size = 10000  # 经验回放缓冲区大小
    target_update = 10  # 目标网络更新频率
    episodes = 50  # 训练轮次

    # 模型参数
    hidden_dim1 = 512
    hidden_dim2 = 256
    dropout = 0.3

    # 设备配置（强制使用单GPU/CPU，避免多设备梯度问题）
    device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"

    # 新增：模型保存路径
    model_save_dir = "D:\\atpbinding\\reinforcementmodel"  # 每轮模型保存目录


config = Config()
np.random.seed(config.random_seed)
random.seed(config.random_seed)
tf.random.set_seed(config.random_seed)

# 创建模型保存目录
os.makedirs(config.model_save_dir, exist_ok=True)

# 修复：设置TensorFlow单设备运行，避免多GPU梯度分散
if config.device == "GPU":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)


# -------------------------- 2. 数据加载（适配Pickle分片+类型统一） --------------------------
def load_pickle_chunks(chunk_dir):
    """
    加载指定文件夹下所有pickle分片文件
    返回：合并后的features（numpy数组）和labels（numpy数组）
    """
    if not os.path.exists(chunk_dir):
        raise FileNotFoundError(f"文件夹 {chunk_dir} 不存在，请检查路径")

    # 获取所有chunk文件（按数字排序）
    chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".pkl")]
    chunk_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # 按chunk编号排序

    if len(chunk_files) == 0:
        raise ValueError(f"文件夹 {chunk_dir} 中没有找到chunk_*.pkl文件")

    all_features = []
    all_labels = []

    print(f"正在加载 {len(chunk_files)} 个分片文件...")
    for idx, file in enumerate(chunk_files):
        file_path = os.path.join(chunk_dir, file)
        with open(file_path, "rb") as f:
            label_list, feature_list = pickle.load(f)  # 适配 (label, feature) 格式

        # 验证数据格式
        assert len(label_list) == len(feature_list), f"文件 {file} 中label和feature长度不匹配"
        assert len(feature_list[
                       0]) == config.feature_dim, f"特征维度错误（应为{config.feature_dim}，实际为{len(feature_list[0])}"

        all_labels.extend(label_list)
        all_features.extend(feature_list)

        if (idx + 1) % 10 == 0:
            print(f"已加载 {idx + 1}/{len(chunk_files)} 个分片，当前数据量：{len(all_labels)} 个样本")

    # 转换为numpy数组（统一标签类型为int64）
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)  # 与TensorFlow默认类型一致

    # 数据清洗：处理异常值
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    features = np.clip(features, -1.0, 1.0)  # 限制特征范围，提升训练稳定性

    print(f"分片加载完成，总样本数：{len(labels)}")
    print(f"特征形状：{features.shape}，标签形状：{labels.shape}，标签类型：{labels.dtype}")
    return features, labels


def load_data():
    """
    加载训练集和测试集：
    - 若已分开存储：分别加载train_data_dir和test_data_dir的分片
    - 若未分开：加载data_dir的分片，自动划分训练/测试集
    """
    if config.train_data_dir and config.test_data_dir:
        # 情况1：训练集和测试集已分开存储
        print("加载训练集分片...")
        X_train, y_train = load_pickle_chunks(config.train_data_dir)
        print("加载测试集分片...")
        X_test, y_test = load_pickle_chunks(config.test_data_dir)

    else:
        raise ValueError("请设置 train_data_dir+test_data_dir 或 data_dir")

    # 转换为TensorFlow数据集（明确指定数据类型，避免自动转换）
    # 优化：增加cache()减少重复加载开销
    train_ds = tf.data.Dataset.from_tensor_slices((
        tf.convert_to_tensor(X_train, dtype=tf.float32),
        tf.convert_to_tensor(y_train, dtype=config.dtype)
    ))
    train_ds = train_ds.shuffle(buffer_size=10000).batch(config.batch_size).prefetch(tf.data.AUTOTUNE).cache()

    test_ds = tf.data.Dataset.from_tensor_slices((
        tf.convert_to_tensor(X_test, dtype=tf.float32),
        tf.convert_to_tensor(y_test, dtype=config.dtype)
    ))
    test_ds = test_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE).cache()

    return train_ds, test_ds


# -------------------------- 3. 强化学习核心组件（类型统一） --------------------------
class DQNModel(keras.Model):
    """DQN网络（Keras实现，明确激活函数和初始化）"""

    def __init__(self, input_dim, output_dim):
        super(DQNModel, self).__init__()
        # 明确设置kernel_initializer，避免权重初始化问题
        self.dense1 = layers.Dense(config.hidden_dim1, activation="relu",
                                   kernel_initializer=keras.initializers.GlorotUniform())
        self.dropout1 = layers.Dropout(config.dropout)
        self.dense2 = layers.Dense(config.hidden_dim2, activation="relu",
                                   kernel_initializer=keras.initializers.GlorotUniform())
        self.dropout2 = layers.Dropout(config.dropout)
        self.dense3 = layers.Dense(output_dim,
                                   kernel_initializer=keras.initializers.GlorotUniform())  # 输出动作价值

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.dense3(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # 转换为TensorFlow张量（统一动作类型为int64）
        return (
            tf.convert_to_tensor(np.array(states), dtype=tf.float32),
            tf.convert_to_tensor(np.array(actions), dtype=config.dtype),  # 动作类型统一
            tf.convert_to_tensor(np.array(rewards), dtype=tf.float32),
            tf.convert_to_tensor(np.array(next_states), dtype=tf.float32),
            tf.convert_to_tensor(np.array(dones), dtype=tf.float32)
        )

    def __len__(self):
        return len(self.buffer)


def compute_reward(pred_actions, true_labels):
    """
    计算奖励：
    - 正确分类：+1.0
    - 错误分类：-0.5
    - 终端奖励：基于批次准确率
    修复：确保输入张量类型一致
    """
    # 强制转换为统一类型（避免类型不匹配）
    pred_actions = tf.cast(pred_actions, config.dtype)
    true_labels = tf.cast(true_labels, config.dtype)

    correct = tf.cast(tf.equal(pred_actions, true_labels), tf.float32)
    immediate_reward = correct * 1.0 + (1 - correct) * (-0.5)

    # 终端奖励（批次准确率归一化）
    batch_acc = tf.reduce_mean(correct)
    terminal_reward = batch_acc * 2.0

    return immediate_reward, terminal_reward


def find_best_mcc_threshold(probabilities, labels):
    """
    在0到1之间以0.01为步长搜索最优分类阈值，计算对应的MCC
    返回最佳阈值和对应的MCC值
    """
    best_mcc = -1.0
    best_threshold = 0.5  # 默认阈值

    # 生成0到1之间的阈值，步长0.01
    thresholds = np.arange(0.0, 1.01, 0.01)

    for threshold in thresholds:
        # 根据当前阈值生成预测标签
        preds = (probabilities >= threshold).astype(int)
        # 计算MCC
        mcc = matthews_corrcoef(labels, preds)
        # 更新最佳阈值和MCC
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    return best_threshold, best_mcc


# -------------------------- 4. 训练函数（完整修复版） --------------------------
def train_dqn_agent(train_ds, test_ds):
    # 初始化策略网络和目标网络
    policy_net = DQNModel(config.feature_dim, config.num_classes)
    target_net = DQNModel(config.feature_dim, config.num_classes)

    # 同步目标网络权重（确保初始权重一致）
    target_net.set_weights(policy_net.get_weights())
    target_net.trainable = False  # 目标网络仅用于预测，不训练

    # 优化器（使用clipnorm防止梯度爆炸）
    optimizer = keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        clipnorm=1.0  # 梯度裁剪，稳定训练
    )

    # 经验回放缓冲区
    replay_buffer = ReplayBuffer(config.memory_size)

    # 训练记录（增加MCC和时间相关字段）
    train_metrics = []
    test_metrics = []

    # 预计算训练集长度（避免重复计算）
    train_ds_length = len(train_ds)

    for episode in range(config.episodes):
        # 记录每轮开始时间
        episode_start_time = time.time()
        episode_reward = 0.0
        episode_correct = 0
        episode_total = 0

        # 探索率衰减
        config.epsilon = max(config.epsilon_min, config.epsilon * config.epsilon_decay)

        # 遍历训练集
        for batch_idx, (states, labels) in enumerate(train_ds):
            batch_size = tf.shape(states)[0]

            # -------------------------- 选择动作（ε-greedy+类型统一） --------------------------
            if random.random() < config.epsilon:
                # 探索：随机选择动作（统一为int64类型）
                actions = tf.random.uniform(
                    shape=(batch_size,),
                    minval=0,
                    maxval=config.num_classes,
                    dtype=config.dtype  # 类型统一
                )
            else:
                # 利用：选择价值最高的动作（关闭训练模式）
                q_values = policy_net(states, training=False)
                actions = tf.argmax(q_values, axis=1)
                actions = tf.cast(actions, config.dtype)  # 类型统一

            # -------------------------- 计算奖励（已处理类型统一） --------------------------
            immediate_rewards, terminal_reward = compute_reward(actions, labels)
            # 只有最后一个batch添加终端奖励
            is_last_batch = (batch_idx == train_ds_length - 1)
            total_rewards = immediate_rewards + (terminal_reward if is_last_batch else 0.0)

            # -------------------------- 存储经验 --------------------------
            dones = tf.ones(batch_size, dtype=tf.float32) if is_last_batch else tf.zeros(batch_size, dtype=tf.float32)
            # 转换为numpy数组存入缓冲区（保持类型一致）
            states_np = states.numpy()
            actions_np = actions.numpy().astype(np.int64)  # 统一类型
            rewards_np = total_rewards.numpy()
            dones_np = dones.numpy()

            # 批量添加经验（优化循环效率）
            new_experiences = [
                (states_np[i], actions_np[i], rewards_np[i], states_np[i], dones_np[i])
                for i in range(batch_size)
            ]
            replay_buffer.buffer.extend(new_experiences)

            # -------------------------- 经验回放训练（稳定梯度计算） --------------------------
            if len(replay_buffer) >= config.batch_size:
                # 采样经验
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(
                    config.batch_size)

                # 所有网络计算都在GradientTape内，确保梯度追踪
                with tf.GradientTape() as tape:
                    # 1. 计算当前Q值（policy_net，开启训练模式）
                    current_q = policy_net(batch_states, training=True)
                    # 提取对应动作的Q值（使用one_hot编码，确保梯度可追踪）
                    batch_actions_one_hot = tf.one_hot(batch_actions, depth=config.num_classes)
                    current_q = tf.reduce_sum(current_q * batch_actions_one_hot, axis=1)

                    # 2. 计算目标Q值（target_net，关闭训练模式）
                    next_q = target_net(batch_next_states, training=False)
                    next_q_max = tf.reduce_max(next_q, axis=1)
                    target_q = batch_rewards + (1 - batch_dones) * config.gamma * next_q_max

                    # 3. 计算损失（MSE）
                    loss = tf.reduce_mean(tf.square(current_q - target_q))

                # 计算并应用梯度
                gradients = tape.gradient(loss, policy_net.trainable_variables)
                # 过滤掉None梯度（防止报错）
                grads_and_vars = [(g, v) for g, v in zip(gradients, policy_net.trainable_variables) if g is not None]
                if grads_and_vars:
                    optimizer.apply_gradients(grads_and_vars)
                # else:
                #     print(f"Warning: 批次 {batch_idx} 未计算到有效梯度，跳过该批次")

            # -------------------------- 记录指标 --------------------------
            # 转换为统一类型计算准确率
            actions_int = tf.cast(actions, tf.int64)
            labels_int = tf.cast(labels, tf.int64)
            correct_count = tf.reduce_sum(tf.cast(tf.equal(actions_int, labels_int), tf.int32)).numpy()

            episode_reward += tf.reduce_sum(total_rewards).numpy()
            episode_correct += correct_count
            episode_total += batch_size

        # -------------------------- 更新目标网络 --------------------------
        if (episode + 1) % config.target_update == 0:
            target_net.set_weights(policy_net.get_weights())
            print(f"Episode {episode + 1}: 更新目标网络权重")

        # -------------------------- 保存每轮模型 --------------------------
        model_save_path = os.path.join(config.model_save_dir, f"dqn_model_episode_{episode + 1}")
        policy_net.save(model_save_path)
        print(f"Episode {episode + 1}: 模型已保存至 {model_save_path}")

        # -------------------------- 训练指标计算 --------------------------
        train_acc = episode_correct / episode_total if episode_total > 0 else 0.0
        train_avg_reward = episode_reward / episode_total if episode_total > 0 else 0.0
        # 计算本轮训练时间
        episode_time = time.time() - episode_start_time
        train_metrics.append({
            "episode": episode + 1,
            "acc": train_acc,
            "avg_reward": train_avg_reward,
            "time_seconds": episode_time
        })

        # -------------------------- 测试集评估（增加MCC计算） --------------------------
        test_acc, test_auc, test_cm, best_threshold, best_mcc = evaluate_agent(policy_net, test_ds)
        test_metrics.append({
            "episode": episode + 1,
            "acc": test_acc,
            "auc": test_auc,
            "best_threshold": best_threshold,
            "best_mcc": best_mcc
        })

        # -------------------------- 打印日志（增加时间信息） --------------------------
        print(f"Episode [{episode + 1}/{config.episodes}] | "
              f"耗时: {episode_time:.2f}秒 | "
              f"Train Acc: {train_acc:.4f} | "
              f"Train Avg Reward: {train_avg_reward:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Test AUC: {test_auc:.4f} | "
              f"Best MCC: {best_mcc:.4f} | "
              f"Epsilon: {config.epsilon:.4f}")

    return policy_net, train_metrics, test_metrics


# -------------------------- 5. 评估函数（增加MCC计算） --------------------------
def evaluate_agent(model, test_ds):
    model.trainable = False  # 切换评估模式
    all_preds = []
    all_probs = []
    all_labels = []

    for states, labels in test_ds:
        # 预测动作价值
        q_values = model(states, training=False)
        # 预测类别（价值最高）
        preds = tf.argmax(q_values, axis=1)
        preds = tf.cast(preds, config.dtype).numpy()  # 类型统一
        # 正类概率（softmax转换）
        probs = tf.nn.softmax(q_values, axis=1)[:, 1].numpy()

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

    # 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    # 计算最佳阈值和对应的MCC
    best_threshold, best_mcc = find_best_mcc_threshold(all_probs, all_labels)

    model.trainable = True  # 恢复训练模式
    return acc, auc, cm, best_threshold, best_mcc


# -------------------------- 6. 主函数（更新最终评估） --------------------------
if __name__ == "__main__":
    # 1. 配置数据路径（请根据你的实际路径修改！）
    # 方式1：训练集和测试集分开存储
    config.train_data_dir = "D:\\atpbinding\\atp227\\inputfile"  # 你的训练集分片文件夹
    config.test_data_dir = "D:\\atpbinding\\atp17\\inputfile"  # 你的测试集分片文件夹

    # 方式2：训练/测试未分开，自动划分（注释上面两行，启用下面一行）
    # config.data_dir = "./all_chunks"  # 你的所有数据分片文件夹

    # 2. 加载数据
    print("=" * 50)
    print(f"使用设备：{config.device}")
    print(f"统一数据类型：{config.dtype}")
    print("=" * 50)
    print("开始加载数据...")
    train_ds, test_ds = load_data()
    print(f"训练集批次数量：{len(train_ds)}")
    print(f"测试集批次数量：{len(test_ds)}")

    # 3. 训练模型
    print("=" * 50)
    print("开始训练DQN模型...")
    model, train_metrics, test_metrics = train_dqn_agent(train_ds, test_ds)

    # 4. 保存结果
    print("=" * 50)
    print("保存最终模型和训练结果...")
    # 保存最终模型
    model.save("final_atp_binding_dqn_model")

    # 保存训练/测试指标（包含时间和MCC相关数据）
    pd.DataFrame(train_metrics).to_csv("train_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(test_metrics).to_csv("test_metrics.csv", index=False, encoding="utf-8")

    # 5. 最终评估（包含MCC）
    print("=" * 50)
    print("最终模型评估...")
    final_acc, final_auc, final_cm, final_threshold, final_mcc = evaluate_agent(model, test_ds)
    print(f"最终测试准确率：{final_acc:.4f}")
    print(f"最终测试AUC：{final_auc:.4f}")
    print(f"最佳分类阈值：{final_threshold:.2f}")
    print(f"最佳MCC：{final_mcc:.4f}")
    print("混淆矩阵：")
    print(final_cm)
    print("=" * 50)
    print("训练完成！")