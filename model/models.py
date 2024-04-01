import torch
import torch.nn as nn


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带mask的softmax交叉熵损失函数"""

    def _sequence_mask(self, X, valid_len, value=0):
        """ 在序列中屏蔽不相关的项。
            接收valid_len是多个有效长度组成的一维tensor,如[1,2]代表第一个序列有效长度为1,第二个序列有效长度为2
        """
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        # 有效长度以外的元素都被置零,不改变原始shape
        return X

    def forward(self, pred, label, valid_len):
        # 不用看标签中的padding的损失
        weights = torch.ones_like(label)
        weights = self._sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)

        # 把整个序列的loss取平均,最后输出的shape是(batch_size)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
