# construct vocab
from rouge import Rouge
import pandas as pd

with open(join(opt.data, 'vocab_cnt.pkl'), 'rb') as f:
    wc = pkl.load(f)
word2idx, idx2word = io.make_vocab(wc, opt.v_size)
opt.word2idx = word2idx
opt.idx2word = idx2word
opt.vocab_size = len(word2idx)
# construct train_data_loader, valid_data_loader

train_data_loader, valid_data_loader = build_loader(
    opt.data, opt.batch_size, word2idx, opt.src_max_len, opt.trg_max_len, opt.batch_workers)

# construct model

overall_model = Seq2SeqModel(opt)
overall_model.to(opt.device)

# construct optimizer
optimizer_ml = torch.optim.Adam(params=filter(
    lambda p: p.requires_grad, overall_model.parameters()), lr=opt.learning_rate)

print(overall_model)
train_model(overall_model, optimizer_ml, train_data_loader,
            valid_data_loader, opt)  # 如果只是想预测，可加载训练好的模型（只保存了模型参数），注释该行。


pd_output = pd.read_csv("./result/submission.csv",
                        sep="\t", names=["index", "output"])
pd_label = pd.read_csv("./datasets/test_label.csv",
                       sep="\t", names=["index", "label"])

output = pd_output.output
label = pd_label.label

rouge = Rouge()
rouge_score = rouge.get_scores(output, label)

rouge_L_f1 = 0
rouge_L_p = 0
rouge_L_r = 0
for d in rouge_score:
    rouge_L_f1 += d["rouge-l"]["f"]
    rouge_L_p += d["rouge-l"]["p"]
    rouge_L_r += d["rouge-l"]["r"]
print("rouge_f1:%.2f" % (rouge_L_f1 / len(rouge_score)))
print("rouge_p:%.2f" % (rouge_L_p / len(rouge_score)))
print("rouge_r:%.2f" % (rouge_L_r / len(rouge_score)))
