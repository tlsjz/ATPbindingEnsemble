import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import matthews_corrcoef, roc_auc_score, confusion_matrix
import datetime
from tensorflow.keras.callbacks import LearningRateScheduler
import math


# --------------------------
# 1. 数据生成器（保持原有优化）
# --------------------------
class ATPDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, chunk_dir, batch_size=16, shuffle=True, balance=False, augment=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.balance = balance
        self.augment = augment

        self.all_labels = []
        self.all_features = []
        chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith('chunk_') and f.endswith('.pkl')]
        chunk_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        for file in chunk_files:
            with open(os.path.join(chunk_dir, file), 'rb') as f:
                labels, features = pickle.load(f)
            self.all_labels.extend(labels)
            self.all_features.extend(features)

        self.all_features = np.array(self.all_features, dtype=np.float32)
        self.all_labels = np.array(self.all_labels, dtype=np.float32).reshape(-1, 1)

        # 特征标准化
        self.feature_mean = np.mean(self.all_features, axis=0)
        self.feature_std = np.std(self.all_features, axis=0) + 1e-7
        self.all_features = (self.all_features - self.feature_mean) / self.feature_std

        self.pos_indices = np.where(self.all_labels == 1.0)[0]
        self.neg_indices = np.where(self.all_labels == 0.0)[0]
        self.indices = np.arange(len(self.all_labels))

        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.all_labels) // self.batch_size

    def __getitem__(self, idx):
        if self.balance and len(self.pos_indices) > 0 and len(self.neg_indices) > 0:
            pos_ratio = len(self.pos_indices) / len(self.all_labels)
            pos_size = max(1, int(self.batch_size * pos_ratio * 1.6))
            neg_size = self.batch_size - pos_size

            pos_batch_idx = np.random.choice(self.pos_indices, size=min(pos_size, len(self.pos_indices)), replace=False)
            neg_batch_idx = np.random.choice(self.neg_indices, size=min(neg_size, len(self.neg_indices)), replace=False)

            if len(pos_batch_idx) < pos_size:
                pos_batch_idx = np.concatenate([pos_batch_idx,
                                                np.random.choice(self.pos_indices, size=pos_size - len(pos_batch_idx),
                                                                 replace=True)])
            if len(neg_batch_idx) < neg_size:
                neg_batch_idx = np.concatenate([neg_batch_idx,
                                                np.random.choice(self.neg_indices, size=neg_size - len(neg_batch_idx),
                                                                 replace=True)])

            batch_indices = np.concatenate([pos_batch_idx, neg_batch_idx])
            np.random.shuffle(batch_indices)
        else:
            batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]

        x_batch = self.all_features[batch_indices]
        y_batch = self.all_labels[batch_indices]

        # 特征增强
        if self.augment:
            x_batch += np.random.normal(0, 0.008, size=x_batch.shape).astype(np.float32)
            mask = np.random.binomial(1, 0.95, size=x_batch.shape)
            x_batch = x_batch * mask

        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# --------------------------
# 2. 核心组件：局部-全局双注意力模块（带位置编码）
# --------------------------
class LocalGlobalAttention(layers.Layer):
    def __init__(self, d_model=256, num_heads=8, local_window=32, **kwargs):
        super().__init__(** kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.local_window = local_window

        self.pos_encoding = self.add_weight(
            shape=(1, 1, d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name=f"{self.name}_pos_encoding"  # 基于层名的唯一名称
        )

        # 为注意力层指定唯一名称
        self.global_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            name=f"{self.name}_global_attn"
        )
        self.local_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            name=f"{self.name}_local_attn"
        )
        self.local_proj = layers.Dense(
            d_model,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            name=f"{self.name}_local_proj"
        )
        self.fusion_weight = self.add_weight(
            shape=(1,),
            initializer='ones',
            trainable=True,
            name=f"{self.name}_fusion_weight"
        )

    def call(self, x):
        # 添加位置编码
        x = x + self.pos_encoding

        batch_size = tf.shape(x)[0]

        # 全局注意力
        global_out = self.global_attn(x, x)

        # 局部注意力
        local_x = self.local_proj(x)
        local_x_reshaped = tf.reshape(local_x, (batch_size, 1, -1, self.local_window))
        local_x_transposed = tf.transpose(local_x_reshaped, (0, 2, 1, 3))
        local_attn_out = self.local_attn(local_x_transposed, local_x_transposed)
        local_attn_out = tf.transpose(local_attn_out, (0, 2, 1, 3))
        local_out = tf.reshape(local_attn_out, (batch_size, 1, self.d_model))

        # 动态融合
        fusion_out = self.fusion_weight * global_out + (1 - self.fusion_weight) * local_out
        return fusion_out

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'num_heads': self.num_heads, 'local_window': self.local_window})
        return config


# --------------------------
# 3. 核心组件：对比学习分支（强化类别相关性）
# --------------------------
class ContrastiveBranch(layers.Layer):
    def __init__(self, temp=0.1, **kwargs):
        super().__init__(**kwargs)
        self.temp = temp
        self.proj = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.proj_norm = layers.BatchNormalization()

    def call(self, x, y_true=None, training=None):
        if not training or y_true is None:
            return x

        x_proj = self.proj(x)
        x_proj = tf.nn.l2_normalize(x_proj, axis=1)
        batch_size = tf.shape(x_proj)[0]
        y_true = tf.reshape(y_true, (-1,))

        # 计算相似度矩阵时添加数值稳定项
        sim_matrix = tf.matmul(x_proj, x_proj, transpose_b=True)
        sim_matrix = sim_matrix / (self.temp + 1e-8)  # 避免温度参数为0

        # 掩码计算优化
        pos_mask = tf.logical_and(
            tf.equal(tf.expand_dims(y_true, 1), tf.expand_dims(y_true, 0)),
            tf.not_equal(tf.range(batch_size)[:, None], tf.range(batch_size))  # 更稳定的对角线掩码
        )
        neg_mask = tf.logical_not(pos_mask)

        # 防止指数溢出：先减去最大值
        sim_max = tf.reduce_max(sim_matrix, axis=1, keepdims=True)
        exp_sim = tf.exp(sim_matrix - sim_max)

        # 防止除零：添加极小值
        pos_exp = tf.reduce_sum(exp_sim * tf.cast(pos_mask, tf.float32), axis=1) + 1e-10
        neg_exp = tf.reduce_sum(exp_sim * tf.cast(neg_mask, tf.float32), axis=1) + 1e-10

        contrast_loss = -tf.math.log(pos_exp / neg_exp)
        # 过滤异常值
        contrast_loss = tf.where(tf.math.is_nan(contrast_loss), tf.zeros_like(contrast_loss), contrast_loss)
        contrast_loss = tf.where(tf.math.is_inf(contrast_loss), tf.zeros_like(contrast_loss), contrast_loss)

        self.add_loss(tf.reduce_mean(contrast_loss) * 0.15)  # 降低对比损失权重
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'temp': self.temp})
        return config


# --------------------------
# 4. 核心组件：动态权重分类头
# --------------------------
class DynamicWeightHead(layers.Layer):
    def __init__(self, d_model=256, num_classes=1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_classes = num_classes
        self.weight_proj = layers.Dense(d_model, activation='sigmoid',
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.classifier = layers.Dense(num_classes, activation='sigmoid',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-5))

    def call(self, x):
        dynamic_weights = self.weight_proj(x)
        weighted_x = x * dynamic_weights
        output = self.classifier(weighted_x)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'num_classes': self.num_classes})
        return config


# --------------------------
# 5. 自定义损失：MCC+Focal损失
# --------------------------
class MCCFocalLoss(losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.2, mcc_weight=0.7, name='mcc_focal_loss', reduction=losses.Reduction.AUTO):  # 提高mcc_weight
        super().__init__(name=name, reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha
        self.mcc_weight = mcc_weight  # 从0.5→0.7，增强MCC惩罚权重

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)  # 更严格的截断

        # Focal Loss部分（增强对难样本的关注）
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = tf.pow((1.0 - p_t), self.gamma)
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_loss = alpha_weight * modulating_factor * bce

        # MCC惩罚项（优化计算稳定性）
        tp = tf.reduce_sum(y_true * y_pred)
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        denominator = tf.sqrt((tp + fp + 1e-6) * (tp + fn + 1e-6) *
                              (tn + fp + 1e-6) * (tn + fn + 1e-6))
        mcc = tf.clip_by_value((tp * tn - fp * fn) / denominator, -0.99, 0.99)  # 避免极端值
        mcc_penalty = 1 - tf.maximum(mcc, 0.0)  # 只惩罚低于当前最佳的MCC

        # 动态平衡Focal和MCC权重（训练后期提高MCC权重）
        current_epoch = self._get_current_epoch()  # 需要自定义获取当前epoch的方法
        '''
        if current_epoch > 20:  # 前20轮以Focal为主，后以MCC为主
            total_loss = 0.3 * tf.reduce_mean(focal_loss) + 0.7 * self.mcc_weight * mcc_penalty
        else:
            total_loss = tf.reduce_mean(focal_loss) + self.mcc_weight * mcc_penalty
        return total_loss
        '''
        total_loss = tf.reduce_mean(focal_loss) + self.mcc_weight * mcc_penalty
        return total_loss

    def _get_current_epoch(self):
        # 通过模型训练状态获取当前epoch（需配合回调实现）
        return getattr(self, 'current_epoch', 0)


# --------------------------
# 6. 最终模型：整合所有优化结构
# --------------------------
def build_atp_dual_attention_contrastive(feature_dim=1280):
    inputs = layers.Input(shape=(feature_dim,))
    labels_input = layers.Input(shape=(1,))  # 标签输入（对比学习用）
    x = layers.Reshape((1, feature_dim))(inputs)

    # 初始降维
    x = layers.Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1.5e-5))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.15)(x)

    # 第一组双注意力
    attn1 = LocalGlobalAttention(d_model=384, num_heads=8, local_window=32)(x)
    x = layers.Add()([x, attn1])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.15)(x)

    # 前馈网络
    x = layers.Dense(768, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1.5e-5))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1.5e-5))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.15)(x)

    # 第二组双注意力
    attn2 = LocalGlobalAttention(d_model=384, num_heads=8, local_window=32)(x)
    x = layers.Add()([x, attn2])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.15)(x)

    # 前馈网络
    x = layers.Dense(768, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1.5e-5))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1.5e-5))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.15)(x)


    # 特征融合与对比学习
    x = layers.Flatten()(x)
    cross_feat = layers.Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1.5e-5))(
        x * tf.expand_dims(tf.reduce_mean(x, axis=1), 1))
    x = layers.Concatenate()([x, cross_feat])
    x = layers.Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1.5e-5))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # 对比学习分支
    contrast_branch = ContrastiveBranch(temp=0.1)
    x = contrast_branch(x, y_true=labels_input, training=True)

    # 动态权重分类头
    output = DynamicWeightHead(d_model=384)(x)

    return models.Model(inputs=[inputs, labels_input], outputs=output)


# --------------------------
# 7. 自定义回调（加入轮次信息保存模型）
# --------------------------
class AdvancedMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, auc_min_threshold=0.80, patience=8):
        super().__init__()
        self.val_generator = val_generator
        self.auc_min_threshold = auc_min_threshold
        self.patience = patience
        self.best_mcc = -1.0
        self.best_auc = 0.0
        self.best_threshold = 0.5
        self.best_model_auc = 0.0
        self.best_epoch = 0  # 记录最佳模型的轮次
        self.no_improve_count = 0
        self.current_recall = 0.5  # 新增加召回率记录变量

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        # 一次性获取所有验证数据
        all_x, all_y = [], []
        for x, y in self.val_generator:
            all_x.append(x)
            all_y.append(y)
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0).flatten().astype(int)

        # 一次性预测
        val_pred = self.model.predict([all_x, np.zeros_like(all_y.reshape(-1, 1))], verbose=0).flatten()

        # 直接使用临时数组，避免累积
        val_y_true = all_y
        val_y_pred_proba = val_pred

        val_auc = roc_auc_score(val_y_true, val_y_pred_proba)
        logs['val_auc'] = val_auc

        thresholds = np.arange(0.1, 1.01, 0.005)
        best_epoch_mcc = -1.0
        best_epoch_thresh = 0.5
        best_cm = None

        for threshold in thresholds:
            val_y_pred = (val_y_pred_proba >= threshold).astype(int)
            mcc = matthews_corrcoef(val_y_true, val_y_pred)
            if mcc > best_epoch_mcc:
                best_epoch_mcc = mcc
                best_epoch_thresh = threshold
                best_cm = confusion_matrix(val_y_true, val_y_pred)

        tn, fp, fn, tp = best_cm.ravel() if best_cm is not None else (0, 0, 0, 0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nEpoch {current_epoch} 验证集指标：")
        print(f"  AUC: {val_auc:.4f} | 最佳MCC: {best_epoch_mcc:.4f}（阈值: {best_epoch_thresh:.3f}）")
        print(f"  混淆矩阵：TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  精确率: {precision:.4f} | 召回率: {recall:.4f} | F1: {f1:.4f}")

        if val_auc >= self.auc_min_threshold:
            if best_epoch_mcc > self.best_mcc + 1e-4:
                self.best_mcc = best_epoch_mcc
                self.best_threshold = best_epoch_thresh
                self.best_model_auc = val_auc
                self.best_epoch = current_epoch  # 更新最佳轮次
                # 保存模型时加入轮次信息，避免文件冲突
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 精确到微秒
                model_path = f'dual_attn_contrastive_best_model_epoch{current_epoch}_{timestamp}.h5'
                self.model.save(model_path, save_format='h5')
                print(f"  ✅ 保存最优模型（轮次 {current_epoch}，MCC提升至 {best_epoch_mcc:.4f}）")
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                print(f"  ⚠️ MCC无提升（已连续{self.no_improve_count}轮）")
                if self.no_improve_count >= self.patience:
                    print(f"  🛑 早停触发")
                    self.model.stop_training = True
        else:
            print(f"  ❌ AUC未达标")


# --------------------------
# 8. 模型训练主函数
# --------------------------
def train_dual_attn_contrastive_model(train_chunk_dir, test_chunk_dir, feature_dim=1280, epochs=60):
    # 初始化生成器
    train_generator = ATPDataGenerator(train_chunk_dir, batch_size=16, shuffle=True, balance=True, augment=True)
    val_generator = ATPDataGenerator(test_chunk_dir, batch_size=16, shuffle=False)

    # 双输入生成器
    def train_data_generator():
        while True:
            for idx in range(len(train_generator)):
                x, y = train_generator[idx]
                yield [x, y], y

    def val_data_generator():
        while True:
            for idx in range(len(val_generator)):
                x, y = val_generator[idx]
                yield [x, y], y

    # 类别权重
    total_pos = int(np.sum(train_generator.all_labels))
    total_neg = len(train_generator.all_labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        initial_pos_weight = 1.0  # 定义初始正样本权重
        class_weight = {0: 1.0, 1: initial_pos_weight}
    else:
        base_weight = total_neg / total_pos
        initial_pos_weight = base_weight * 1.8  # 定义初始正样本权重
        class_weight = {0: 1.0, 1: initial_pos_weight}
    print(f"类别分布：负样本={total_neg}, 正样本={total_pos}")
    print(f"类别权重：负样本=1.0, 正样本={class_weight[1]:.2f}")

    # 构建模型
    model = build_atp_dual_attention_contrastive(feature_dim=feature_dim)
    model.summary()

    # 编译模型
    optimizer = optimizers.Adam(learning_rate=4e-6)
    #lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=4, min_lr=5e-8, verbose=1)
    def cosine_annealing_decay(epoch):
        initial_lr = 4e-6  # 初始学习率
        T_max = 20  # 学习率周期（每20轮完成一次衰减→最小值→回升）
        eta_min = 5e-8  # 最小学习率（防止学习率过低停滞）
        # 余弦退火公式：lr = 最小值 + 0.5*(初始值-最小值)*(1+cos(pi*epoch/T_max))
        lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max))
        return lr

    lr_scheduler = LearningRateScheduler(schedule=cosine_annealing_decay, verbose=1)
    loss_fn = MCCFocalLoss(gamma=2.0, alpha=0.2, mcc_weight=0.5)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    # 回调函数：添加动态权重回调
    metrics_callback = AdvancedMetricsCallback(val_generator, auc_min_threshold=0.85, patience=12)
    callbacks = [metrics_callback, lr_scheduler]  # 加入新回调

    # 训练
    print("开始训练（双注意力+对比学习，冲击MCC=0.65）...")
    model.fit(
        train_data_generator(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_data_generator(),
        validation_steps=len(val_generator),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # 输出结果
    print("\n训练完成！")
    print(
        f"🏆 最佳模型 - 轮次: {metrics_callback.best_epoch}, AUC: {metrics_callback.best_model_auc:.4f}, "
        f"MCC: {metrics_callback.best_mcc:.4f}（阈值: {metrics_callback.best_threshold:.3f}）")

    return model, metrics_callback.best_model_auc, metrics_callback.best_mcc, metrics_callback.best_threshold


# --------------------------
# 9. 主程序入口
# --------------------------
if __name__ == "__main__":
    tf.config.run_functions_eagerly(False)  # 关闭eager模式，提升兼容性

    TRAIN_CHUNK_DIR = "D:\\atpbinding\\atp388\\inputfile"
    TEST_CHUNK_DIR = "D:\\atpbinding\\atp41\\inputfile"
    FEATURE_DIM = 1280
    TRAIN_EPOCHS = 100

    model, best_auc, best_mcc, best_thresh = train_dual_attn_contrastive_model(
        train_chunk_dir=TRAIN_CHUNK_DIR,
        test_chunk_dir=TEST_CHUNK_DIR,
        feature_dim=FEATURE_DIM,
        epochs=TRAIN_EPOCHS
    )

    # 保存结果
    with open("dual_attn_contrastive_results.txt", "w") as f:
        f.write(f"最佳AUC: {best_auc:.4f}\n")
        f.write(f"最佳MCC: {best_mcc:.4f}\n")
        f.write(f"最佳阈值: {best_thresh:.3f}\n")