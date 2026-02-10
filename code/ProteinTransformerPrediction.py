import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import matthews_corrcoef, roc_auc_score, confusion_matrix, accuracy_score
import warnings
from datetime import datetime  # 新增：用于生成准确的预测时间

warnings.filterwarnings('ignore')
from contractivetransformer import (
    ATPDataGenerator,
    LocalGlobalAttention,
    ContrastiveBranch,
    DynamicWeightHead,
    MCCFocalLoss
)

# 自定义对象映射（加载模型时必须指定）
custom_objects = {
    'LocalGlobalAttention': LocalGlobalAttention,
    'ContrastiveBranch': ContrastiveBranch,
    'DynamicWeightHead': DynamicWeightHead,
    'MCCFocalLoss': MCCFocalLoss
}


# --------------------------
# 2. 核心预测函数
# --------------------------
def predict_protein_atp_binding(model_path, input_dir, output_dir, feature_dim=1280):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects
    )
    print(f"✅ 成功加载模型: {model_path}")
    print("模型加载完成！")

    # 获取所有输入文件
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.pickle')]
    print(f"\n找到 {len(input_files)} 个蛋白质文件待预测")

    # 遍历每个文件进行预测
    for idx, file_name in enumerate(input_files, 1):
        print(f"\n[{idx}/{len(input_files)}] 处理文件: {file_name}")

        # 加载数据
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'rb') as f:
            labels, features = pickle.load(f)

        # 数据预处理
        labels = np.array(labels, dtype=np.float32).reshape(-1, 1)
        features = np.array(features, dtype=np.float32)

        # 特征标准化（使用训练数据的均值和标准差，这里使用当前数据的统计量）
        feature_mean = np.mean(features, axis=0)
        feature_std = np.std(features, axis=0) + 1e-7
        features = (features - feature_mean) / feature_std

        # 模型预测
        # 模型需要双输入，第二个输入为标签（预测时用0填充）
        pred_proba = model.predict([features, np.zeros_like(labels)], verbose=0).flatten()

        # 计算最优阈值（遍历寻找最优MCC的阈值）
        thresholds = np.arange(0.1, 1.01, 0.005)
        best_mcc = -1.0
        best_threshold = 0.5
        best_cm = None

        for threshold in thresholds:
            pred_labels = (pred_proba >= threshold).astype(int)
            mcc = matthews_corrcoef(labels.flatten(), pred_labels)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold
                best_cm = confusion_matrix(labels.flatten(), pred_labels)

        # 使用最优阈值计算预测标签
        pred_labels_opt = (pred_proba >= best_threshold).astype(int)

        # 计算各项指标
        # 基础指标
        tn, fp, fn, tp = best_cm.ravel() if best_cm is not None else (0, 0, 0, 0)
        acc = accuracy_score(labels.flatten(), pred_labels_opt)
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 灵敏度/召回率
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 特异度
        mcc = best_mcc

        # AUC计算
        try:
            auc = roc_auc_score(labels.flatten(), pred_proba)
        except:
            auc = 0.0

        # 打印当前文件的预测结果
        print(f"  最优阈值: {best_threshold:.3f}")
        print(f"  准确率(ACC): {acc:.4f} | 灵敏度(SEN): {sen:.4f} | 特异度(SPE): {spe:.4f}")
        print(f"  MCC: {mcc:.4f} | AUC: {auc:.4f}")
        print(f"  混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # 保存预测结果（确保pred_proba完整保存）
        output_file = os.path.join(output_dir, file_name)
        result_data = {
            'true_labels': labels.flatten(),
            'pred_proba': pred_proba,  # 明确保存预测概率
            'pred_labels_opt': pred_labels_opt,
            'best_threshold': best_threshold,
            'metrics': {
                'accuracy': acc,
                'sensitivity': sen,
                'specificity': spe,
                'mcc': mcc,
                'auc': auc,
                'confusion_matrix': {
                    'tn': tn,
                    'fp': fp,
                    'fn': fn,
                    'tp': tp
                }
            }
        }

        # 保存pickle文件（确保pred_proba被序列化）
        with open(output_file, 'wb') as f:
            pickle.dump(result_data, f, protocol=pickle.HIGHEST_PROTOCOL)  # 使用高协议确保兼容性
        print(f"  预测结果（含pred_proba）已保存至: {output_file}")

        # 保存详细的指标报告（文本文件）
        report_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"蛋白质文件: {file_name}\n")
            f.write(f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # 修复时间显示
            f.write("=" * 50 + "\n")
            f.write(f"最优阈值: {best_threshold:.3f}\n")
            f.write("\n【性能指标】\n")
            f.write(f"准确率 (ACC): {acc:.4f}\n")
            f.write(f"灵敏度 (SEN/Recall): {sen:.4f}\n")
            f.write(f"特异度 (SPE): {spe:.4f}\n")
            f.write(f"马修斯相关系数 (MCC): {mcc:.4f}\n")
            f.write(f"ROC曲线下面积 (AUC): {auc:.4f}\n")
            f.write("\n【混淆矩阵】\n")
            f.write(f"真阴性 (TN): {tn}\n")
            f.write(f"假阳性 (FP): {fp}\n")
            f.write(f"假阴性 (FN): {fn}\n")
            f.write(f"真阳性 (TP): {tp}\n")
            f.write("\n【预测结果统计】\n")
            f.write(f"总样本数: {len(labels)}\n")
            f.write(f"正样本数: {int(np.sum(labels))}\n")
            f.write(f"负样本数: {len(labels) - int(np.sum(labels))}\n")
            f.write(f"预测正样本数: {int(np.sum(pred_labels_opt))}\n")
            f.write(f"预测负样本数: {len(pred_labels_opt) - int(np.sum(pred_labels_opt))}\n")
            f.write(f"\n【pred_proba统计】\n")
            f.write(f"预测概率均值: {np.mean(pred_proba):.4f}\n")
            f.write(f"预测概率标准差: {np.std(pred_proba):.4f}\n")
            f.write(f"预测概率最小值: {np.min(pred_proba):.4f}\n")
            f.write(f"预测概率最大值: {np.max(pred_proba):.4f}\n")

    print(f"\n所有预测完成！结果已保存至: {output_dir}")


# --------------------------
# 3. 主程序入口
# --------------------------
if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "D:\\atpbinding\\atp388\\dual_attn_contrastive_best_model_epoch54_20251202_190346_673948.h5"
    INPUT_DIR = "D:\\atpbinding\\atp41\\proteininput"
    OUTPUT_DIR = "D:\\atpbinding\\atp41\\proteintransformerprediction"  # 指定保存pred_proba的目录
    FEATURE_DIM = 1280  # 需与训练时一致

    # 执行预测
    predict_protein_atp_binding(
        model_path=MODEL_PATH,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        feature_dim=FEATURE_DIM
    )