import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef
import warnings

warnings.filterwarnings('ignore')


# -------------------------- 配置参数 --------------------------
class PredictConfig:
    # 模型路径
    model_path = "D:\\atpbinding\\atp227\\dqn_model_episode_45"
    # model_path = "D:\\atpbinding\\atp388\\dqn_model_episode_12"
    # 待预测文件目录
    input_dir = "D:\\atpbinding\\atp41\\proteininput"
    # 预测结果保存目录
    output_dir = "D:\\atpbinding\\atp41\\proteindqnprediction"
    feature_dim = 1280  # 必须与训练时一致
    num_classes = 2
    dtype = np.int64  # NumPy兼容类型
    tf_dtype = tf.int64  # TensorFlow类型


config = PredictConfig()

# 创建输出目录
os.makedirs(config.output_dir, exist_ok=True)

# 设置TensorFlow设备
if tf.config.list_physical_devices('GPU'):
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"使用GPU: {gpus[0]}")
else:
    print("使用CPU进行预测")


# -------------------------- 定义模型结构 --------------------------
class DQNModel(keras.Model):
    """DQN网络结构（必须与训练代码完全一致）"""

    def __init__(self, input_dim, output_dim):
        super(DQNModel, self).__init__()
        self.dense1 = layers.Dense(512, activation="relu",
                                   kernel_initializer=keras.initializers.GlorotUniform())
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(256, activation="relu",
                                   kernel_initializer=keras.initializers.GlorotUniform())
        self.dropout2 = layers.Dropout(0.3)
        self.dense3 = layers.Dense(output_dim,
                                   kernel_initializer=keras.initializers.GlorotUniform())

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.dense3(x)


# -------------------------- 工具函数 --------------------------
def parse_confusion_matrix(cm):
    """
    解析混淆矩阵，返回TP/TN/FP/FN四个明确指标
    cm: 混淆矩阵（sklearn生成，labels=[0,1]）
    返回：tn, fp, fn, tp
    """
    # 确保混淆矩阵维度正确
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        # 只有一类的特殊情况
        if np.unique(cm)[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    # 转换为整数（避免浮点型）
    return int(tn), int(fp), int(fn), int(tp)


def find_best_mcc_threshold(probabilities, labels):
    """
    搜索最优分类阈值，返回：最佳阈值、最佳MCC、包含TP/TN/FP/FN的完整指标字典
    """
    best_mcc = -1.0
    best_threshold = 0.5
    best_metrics = {}

    thresholds = np.arange(0.0, 1.01, 0.01)

    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(config.dtype)
        acc = accuracy_score(labels, preds)

        # 计算混淆矩阵并解析TP/TN/FP/FN
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = parse_confusion_matrix(cm)

        # 计算核心指标
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 灵敏度 (TP/(TP+FN))
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 特异度 (TN/(TN+FP))
        mcc = matthews_corrcoef(labels, preds)

        # 更新最佳指标（包含TP/TN/FP/FN）
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
            best_metrics = {
                'accuracy': acc,
                'sensitivity': sen,
                'specificity': spe,
                'mcc': mcc,
                'confusion_matrix': cm,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp,
                'threshold': threshold
            }

    return best_threshold, best_mcc, best_metrics


def load_single_pickle(file_path):
    """加载单个非分片pickle文件"""
    with open(file_path, "rb") as f:
        label_list, feature_list = pickle.load(f)

    features = np.array(feature_list, dtype=np.float32)
    labels = np.array(label_list, dtype=config.dtype)

    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    features = np.clip(features, -1.0, 1.0)

    assert len(features[0]) == config.feature_dim, f"特征维度错误，应为{config.feature_dim}"

    return labels, features


# -------------------------- 单文件预测函数 --------------------------
def predict_single_file(model, file_path):
    """单文件预测，输出明确的TP/TN/FP/FN指标"""
    # 1. 加载数据
    true_labels, features = load_single_pickle(file_path)
    file_name = os.path.basename(file_path)
    print(f"\n处理文件 {file_name}: 样本数={len(true_labels)}")

    # 2. 模型预测
    model.trainable = False
    test_states = tf.convert_to_tensor(features, dtype=tf.float32)
    q_values = model(test_states, training=False)
    probs = tf.nn.softmax(q_values, axis=1)[:, 1].numpy()
    default_preds = (probs >= 0.5).astype(config.dtype)

    # 3. 计算AUC
    try:
        auc = roc_auc_score(true_labels, probs)
    except ValueError:
        auc = np.nan
        print(f"警告：{file_name} 标签只有一类，无法计算AUC")

    # 4. 寻找最优阈值并获取完整指标
    best_threshold, best_mcc, best_metrics = find_best_mcc_threshold(probs, true_labels)

    # 5. 整理并保存结果（包含TP/TN/FP/FN）
    output_path = os.path.join(config.output_dir, file_name)
    results = {
        'true_labels': true_labels.tolist(),
        'pred_probs': probs.tolist(),
        'default_preds': default_preds.tolist(),
        'best_threshold_preds': (probs >= best_threshold).astype(config.dtype).tolist(),
        'auc': auc,
        'best_metrics': best_metrics,
        # 单独保存混淆矩阵拆解结果（便于查看）
        'confusion_matrix_parsed': {
            'tn': best_metrics['tn'],
            'fp': best_metrics['fp'],
            'fn': best_metrics['fn'],
            'tp': best_metrics['tp']
        }
    }
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  预测结果已保存至: {output_path}")

    # 6. 整理性能指标（包含TP/TN/FP/FN）
    metrics = {
        'filename': file_name,
        'sample_count': len(true_labels),
        'auc': auc,
        'best_threshold': best_threshold,
        'accuracy': best_metrics['accuracy'],
        'sensitivity': best_metrics['sensitivity'],
        'specificity': best_metrics['specificity'],
        'mcc': best_metrics['mcc'],
        # 明确保存TP/TN/FP/FN
        'tn': best_metrics['tn'],
        'fp': best_metrics['fp'],
        'fn': best_metrics['fn'],
        'tp': best_metrics['tp'],
        # 保留原始混淆矩阵（便于核对）
        'confusion_matrix': best_metrics['confusion_matrix'].tolist()
    }

    # 打印优化后的指标（明确展示TP/TN/FP/FN）
    print(f"  最优阈值: {metrics['best_threshold']:.2f}")
    print(f"  准确率(ACC): {metrics['accuracy']:.4f}")
    print(f"  灵敏度(Sen): {metrics['sensitivity']:.4f}")
    print(f"  特异度(Spe): {metrics['specificity']:.4f}")
    print(f"  MCC: {metrics['mcc']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}" if not np.isnan(metrics['auc']) else "  AUC: N/A")
    # 重点：明确输出TP/TN/FP/FN
    print(f"  混淆矩阵拆解:")
    print(f"    真阴性(TN): {metrics['tn']}")
    print(f"    假阳性(FP): {metrics['fp']}")
    print(f"    假阴性(FN): {metrics['fn']}")
    print(f"    真阳性(TP): {metrics['tp']}")

    return metrics


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 加载模型
    print(f"加载模型：{config.model_path}")
    model = tf.keras.models.load_model(config.model_path)
    model.trainable = False
    print("模型加载完成！")

    # 批量处理文件
    pickle_files = [f for f in os.listdir(config.input_dir) if f.endswith(".pickle")]
    if not pickle_files:
        raise ValueError(f"输入目录 {config.input_dir} 中未找到pickle文件")

    print(f"\n找到 {len(pickle_files)} 个待预测文件")

    all_metrics = []
    for idx, file_name in enumerate(pickle_files, 1):
        file_path = os.path.join(config.input_dir, file_name)
        try:
            metrics = predict_single_file(model, file_path)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"处理 {file_name} 时出错: {str(e)}")
            continue

    # 保存汇总指标（CSV包含TP/TN/FP/FN）
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        summary_path = os.path.join(config.output_dir, "prediction_summary.csv")
        metrics_df.to_csv(summary_path, index=False, encoding="utf-8")
        print(f"\n所有文件处理完成！汇总指标已保存至: {summary_path}")

        # 打印整体统计（包含TP/TN/FP/FN汇总）
        print("\n=== 整体统计 ===")
        valid_auc = [m['auc'] for m in all_metrics if not np.isnan(m['auc'])]
        print(f"有效AUC文件数: {len(valid_auc)}/{len(all_metrics)}")
        if valid_auc:
            print(f"AUC均值: {np.mean(valid_auc):.4f}")
        print(f"平均准确率: {np.mean([m['accuracy'] for m in all_metrics]):.4f}")
        print(f"平均MCC: {np.mean([m['mcc'] for m in all_metrics]):.4f}")

        # 汇总TP/TN/FP/FN
        total_tn = sum([m['tn'] for m in all_metrics])
        total_fp = sum([m['fp'] for m in all_metrics])
        total_fn = sum([m['fn'] for m in all_metrics])
        total_tp = sum([m['tp'] for m in all_metrics])
        print(f"汇总混淆矩阵:")
        print(f"  总真阴性(TN): {total_tn}")
        print(f"  总假阳性(FP): {total_fp}")
        print(f"  总假阴性(FN): {total_fn}")
        print(f"  总真阳性(TP): {total_tp}")
    else:
        print("\n无有效预测结果！")