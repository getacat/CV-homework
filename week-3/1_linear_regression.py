import numpy as np
import random

# 用numpy的方法替代了for，不知道这算不算是’python way'？
# 效果，在增大batch_size时，效率明显提高
# 参数维度
# w_b: (2,1)
# x: (num, 2)
# y: (num,1)
# pred_y: (num,1)
# d: (2,1)

def inference(w_b, x):
    pred_y = x @ w_b
    return pred_y

def eval_loss(batch_x, batch_gt_y, w_b):
    pred_y = inference(w_b, batch_x)
    l = np.square(pred_y - batch_gt_y)
    l = np.mean(l)
    return l

# 计算新的w, b
def next(batch_x, batch_gt_y, w_b, lr):
    pred_y = inference(w_b, batch_x)
    d = batch_x * (pred_y - batch_gt_y) * lr
    d = np.mean(d, axis=0).reshape(2,1)
    w_b -= d
    return w_b

def train(x, gt_y, batch_size, lr, max_iter):
    w_b = np.zeros((2, 1))
    for i in range(max_iter):
        index = np.random.randint(x.shape[0], size = batch_size)
        batch_x = x[index]
        batch_gt_y = gt_y[index]
        pred_w_b = next(batch_x, batch_gt_y, w_b, lr)
        loss = eval_loss(batch_x, batch_gt_y, w_b)
    return pred_w_b, loss

# 随机形成w,b和num组点
def gen_sample_data(num):
    w_b = np.random.uniform(1, 10, (2, 1))
    x = np.random.uniform(0, 100, (num, 2))
    x[:, 1] = 10
    y = x@w_b + np.random.uniform(-1, 1)
    return w_b, x, y

def run():
    lr = 0.0001
    num_samples = 10000
    max_iter = 1000
    batch_size = 5000
    w_b, x, y = gen_sample_data(num_samples)
    pred_w_b, loss = train(x, y, batch_size, lr, max_iter)
    print('w,b:',w_b.flatten(), 'pred:',pred_w_b.flatten())
    print(loss)

if __name__ == '__main__':
    run()
    # t = %timeit -o run()