import numpy as np
import pickle
import os
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef
import warnings

warnings.filterwarnings('ignore')


# -------------------------- 配置参数 --------------------------
class EnsembleConfig:
    # 两个基分类器的结果路径
    classifier1_result_path = "D:\\atpbinding\\atp41\\proteintransformerprediction\\5E84_F.pickle"  # 第一个分类器结果
    classifier2_result_path = "D:\\atpbinding\\atp41\\proteindqnprediction\\5E84_F.pickle"  # 第二个分类器结果
    # 集成结果保存路径
    ensemble_output_path = "D:\\atpbinding\\atp41\\proteinensembleprediction\\5E84_F_ensemble.pickle"
    # 权重搜索范围（步长越小，精度越高，但耗时越长）
    weight_step = 0.05  # 权重搜索步长，0.05表示搜索0.0,0.05,0.1,...,1.0
    # 阈值搜索范围
    threshold_step = 0.01


# 初始化配置
config = EnsembleConfig()
# 创建集成结果保存目录
os.makedirs(os.path.dirname(config.ensemble_output_path), exist_ok=True)


# -------------------------- 核心工具函数 --------------------------
def load_classifier_result(file_path):
    """
    加载分类器预测结果，统一数据格式
    返回：true_labels (np.array), pred_probs (np.array)
    """
    with open(file_path, 'rb') as f:
        result = pickle.load(f)

    # 适配两个分类器的不同格式
    if 'true_labels' in result:
        true_labels = np.array(result['true_labels'])
    else:
        raise ValueError("结果文件中未找到true_labels字段")

    if 'pred_proba' in result:  # 第一个分类器格式
        pred_probs = np.array(result['pred_proba'])
    elif 'pred_probs' in result:  # 第二个分类器格式
        pred_probs = np.array(result['pred_probs'])
    else:
        raise ValueError("结果文件中未找到预测概率字段（pred_proba/pred_probs）")

    # 验证维度匹配
    assert len(true_labels) == len(pred_probs), \
        f"真实标签({len(true_labels)})和预测概率({len(pred_probs)})长度不匹配"

    # 归一化概率到[0,1]（防止数值异常）
    pred_probs = np.clip(pred_probs, 0.0, 1.0)

    return true_labels, pred_probs


def find_best_threshold(probabilities, true_labels, step=0.01):
    """
    搜索最优分类阈值（基于MCC最大化）
    返回：best_threshold, best_metrics
    """
    best_mcc = -1.0
    best_threshold = 0.5
    best_metrics = {}

    thresholds = np.arange(0.0, 1.0 + step, step)
    for threshold in thresholds:
        pred_labels = (probabilities >= threshold).astype(int)

        # 计算核心指标
        acc = accuracy_score(true_labels, pred_labels)
        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, cm[1, 1])
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 灵敏度
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 特异度
        mcc = matthews_corrcoef(true_labels, pred_labels)

        # 更新最优阈值
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
            best_metrics = {
                'accuracy': acc,
                'sensitivity': sen,
                'specificity': spe,
                'mcc': mcc,
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp),
                'confusion_matrix': cm.tolist()
            }

    # 计算AUC（基于原始概率）
    try:
        auc = roc_auc_score(true_labels, probabilities)
    except ValueError:
        auc = np.nan

    best_metrics['auc'] = auc

    return best_threshold, best_metrics


def search_best_weights(true_labels, prob1, prob2, weight_step=0.05):
    """
    搜索最优加权权重（w1*prob1 + w2*prob2，w1+w2=1）
    返回：best_w1, best_w2, ensemble_probs, best_threshold, best_metrics
    """
    best_mcc = -1.0
    best_w1 = 0.5
    best_w2 = 0.5
    best_ensemble_probs = None
    best_threshold = 0.5
    best_ensemble_metrics = {}

    # 搜索权重范围：w1从0到1，步长weight_step
    w1_list = np.arange(0.0, 1.0 + weight_step, weight_step)

    print("开始搜索最优权重组合...")
    for idx, w1 in enumerate(w1_list):
        w2 = 1.0 - w1

        # 计算集成概率
        ensemble_probs = w1 * prob1 + w2 * prob2

        # 寻找当前权重下的最优阈值
        threshold, metrics = find_best_threshold(ensemble_probs, true_labels, config.threshold_step)

        # 打印进度
        if idx % 5 == 0:  # 每5个权重打印一次
            print(f"  权重组合 w1={w1:.2f}, w2={w2:.2f} | MCC={metrics['mcc']:.4f} | ACC={metrics['accuracy']:.4f}")

        # 更新最优权重
        if metrics['mcc'] > best_mcc:
            best_mcc = metrics['mcc']
            best_w1 = w1
            best_w2 = w2
            best_ensemble_probs = ensemble_probs.copy()
            best_threshold = threshold
            best_ensemble_metrics = metrics

    return best_w1, best_w2, best_ensemble_probs, best_threshold, best_ensemble_metrics


# -------------------------- 集成主函数 --------------------------
def ensemble_classifiers():
    """
    加权集成两个分类器的预测结果
    """
    # 1. 加载两个分类器的结果
    print("加载第一个分类器结果...")
    true_labels1, prob1 = load_classifier_result(config.classifier1_result_path)
    print("加载第二个分类器结果...")
    true_labels2, prob2 = load_classifier_result(config.classifier2_result_path)

    # 验证两个分类器的真实标签一致
    assert np.array_equal(true_labels1, true_labels2), \
        "两个分类器的真实标签不一致，无法集成"
    true_labels = true_labels1  # 统一使用第一个分类器的真实标签

    print(f"\n数据加载完成：")
    print(f"  氨基酸总数: {len(true_labels)}")
    print(f"  正样本数(ATP结合): {np.sum(true_labels)}")
    print(f"  负样本数(非ATP结合): {len(true_labels) - np.sum(true_labels)}")

    # 2. 搜索最优权重
    print("\n" + "=" * 50)
    best_w1, best_w2, ensemble_probs, best_threshold, ensemble_metrics = \
        search_best_weights(true_labels, prob1, prob2, config.weight_step)

    # 3. 生成集成后的预测标签
    ensemble_pred_labels = (ensemble_probs >= best_threshold).astype(int)

    # 4. 整理集成结果
    ensemble_result = {
        # 基础信息
        'protein_id': os.path.basename(config.classifier1_result_path).split('.')[0],
        'total_amino_acids': len(true_labels),
        # 最优权重
        'best_weights': {
            'classifier1_weight': best_w1,
            'classifier2_weight': best_w2
        },
        # 真实标签
        'true_labels': true_labels.tolist(),
        # 各分类器原始概率
        'classifier1_pred_probs': prob1.tolist(),
        'classifier2_pred_probs': prob2.tolist(),
        # 集成后结果
        'ensemble_pred_probs': ensemble_probs.tolist(),  # 每个氨基酸的集成概率
        'best_threshold': best_threshold,
        'ensemble_pred_labels': ensemble_pred_labels.tolist(),
        # 性能指标
        'metrics': {
            'accuracy': ensemble_metrics['accuracy'],
            'sensitivity': ensemble_metrics['sensitivity'],
            'specificity': ensemble_metrics['specificity'],
            'mcc': ensemble_metrics['mcc'],
            'auc': ensemble_metrics['auc'],
            'confusion_matrix': {
                'tn': ensemble_metrics['tn'],
                'fp': ensemble_metrics['fp'],
                'fn': ensemble_metrics['fn'],
                'tp': ensemble_metrics['tp']
            }
        }
    }

    # 5. 保存集成结果
    with open(config.ensemble_output_path, 'wb') as f:
        pickle.dump(ensemble_result, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 6. 打印最终结果
    print("\n" + "=" * 50)
    print("集成结果汇总")
    print("=" * 50)
    print(f"最优权重组合:")
    print(f"  分类器1权重: {best_w1:.4f}")
    print(f"  分类器2权重: {best_w2:.4f}")
    print(f"\n最优阈值: {best_threshold:.4f}")
    print(f"\n性能指标（最优权重+最优阈值）:")
    print(f"  准确率(ACC): {ensemble_metrics['accuracy']:.4f}")
    print(f"  灵敏度(Sen): {ensemble_metrics['sensitivity']:.4f}")
    print(f"  特异度(Spe): {ensemble_metrics['specificity']:.4f}")
    print(f"  马修斯相关系数(MCC): {ensemble_metrics['mcc']:.4f}")
    print(f"  AUC: {ensemble_metrics['auc']:.4f}" if not np.isnan(ensemble_metrics['auc']) else "  AUC: N/A")
    print(f"\n混淆矩阵详情:")
    print(f"  真阴性(TN): {ensemble_metrics['tn']}")
    print(f"  假阳性(FP): {ensemble_metrics['fp']}")
    print(f"  假阴性(FN): {ensemble_metrics['fn']}")
    print(f"  真阳性(TP): {ensemble_metrics['tp']}")
    print(f"\n集成结果已保存至: {config.ensemble_output_path}")

    # 7. 生成汇总CSV（便于批量分析）
    summary_data = {
        'protein_id': [ensemble_result['protein_id']],
        'classifier1_weight': [best_w1],
        'classifier2_weight': [best_w2],
        'best_threshold': [best_threshold],
        'accuracy': [ensemble_metrics['accuracy']],
        'sensitivity': [ensemble_metrics['sensitivity']],
        'specificity': [ensemble_metrics['specificity']],
        'mcc': [ensemble_metrics['mcc']],
        'auc': [ensemble_metrics['auc']],
        'tn': [ensemble_metrics['tn']],
        'fp': [ensemble_metrics['fp']],
        'fn': [ensemble_metrics['fn']],
        'tp': [ensemble_metrics['tp']]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(os.path.dirname(config.ensemble_output_path), "ensemble_summary.csv")

    # 如果文件已存在，追加数据；否则新建
    if os.path.exists(summary_csv_path):
        summary_df.to_csv(summary_csv_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
    print(f"汇总CSV已保存至: {summary_csv_path}")

    return ensemble_result


# -------------------------- 批量集成函数（可选） --------------------------
def batch_ensemble(classifier1_dir, classifier2_dir, ensemble_output_dir):
    """
    批量处理多个蛋白质序列的集成
    参数：
        classifier1_dir: 第一个分类器结果目录
        classifier2_dir: 第二个分类器结果目录
        ensemble_output_dir: 集成结果保存目录
    """
    # 获取第一个分类器的所有结果文件
    pickle_files = [f for f in os.listdir(classifier1_dir) if f.endswith('.pickle')]
    if not pickle_files:
        raise ValueError(f"第一个分类器目录 {classifier1_dir} 中未找到pickle文件")

    os.makedirs(ensemble_output_dir, exist_ok=True)

    for file_name in pickle_files:
        # 构建文件路径
        c1_path = os.path.join(classifier1_dir, file_name)
        c2_path = os.path.join(classifier2_dir, file_name)
        ensemble_path = os.path.join(ensemble_output_dir, file_name)

        # 检查第二个分类器文件是否存在
        if not os.path.exists(c2_path):
            print(f"警告：第二个分类器缺少 {file_name}，跳过")
            continue

        # 更新配置
        config.classifier1_result_path = c1_path
        config.classifier2_result_path = c2_path
        config.ensemble_output_path = ensemble_path

        # 执行集成
        print(f"\n处理蛋白质: {file_name}")
        try:
            ensemble_classifiers()
        except Exception as e:
            print(f"处理 {file_name} 失败: {str(e)}")
            continue


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 方式1：处理单个蛋白质序列
    ensemble_classifiers()

    # 方式2：批量处理多个蛋白质序列（注释方式1，取消注释方式2）
    # classifier1_dir = "D:\\atpbinding\\atp41\\proteintransformerprediction"
    # classifier2_dir = "D:\\atpbinding\\atp41\\proteindqnprediction"
    # ensemble_output_dir = "D:\\atpbinding\\atp41\\ensemble_result"
    # batch_ensemble(classifier1_dir, classifier2_dir, ensemble_output_dir)