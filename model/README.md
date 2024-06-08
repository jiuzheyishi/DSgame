# Transformer's Inplementation

## refer 

1. [nn.labml.ai](https://nn.labml.ai/transformers/)
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)


#### 关于Mask

Encoder中的Padding Mask：用于对输入序列中的padding tokens（填充标记）进行掩蔽。这样做可以确保模型在自注意力层对实际有效的词进行操作，而忽略padding tokens。

Decoder中的Look-Ahead Mask（前瞻掩码）：用于确保在decoder的自注意力层中，位置i的token只能依赖于位置小于i的tokens。这是为了防止在预测序列中的token获取未来位置的信息。在实践中，这通常通过使用一个三角形的矩阵实现，该矩阵的上三角（包含对角线）部分被设置为无限大（或一个非常大的数字），这样经过softmax层时，上三角部分的权重接近于0。

Decoder中的Padding Mask：与Encoder类似，用于掩蔽目标序列中的padding tokens。

Encoder-Decoder Attention Layers中的Padding Mask：在Seq2Seq模型中，Encoder的输出会被用作Decoder的一个输入，在这里面也需要使用Padding Mask来确保Decoder的自注意力层不会考虑到Encoder输出中的无效填充区域。