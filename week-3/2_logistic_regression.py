import numpy as np
import random

# sigmoid函数
def inference(w_b, x):
    pred_y = 1.0 / (np.exp(- x @ w_b) + 1)
    return pred_y

def eval_loss(batch_x, batch_gt_y, w_b):
    pred_y = inference(w_b, batch_x)
    loss = batch_gt_y * np.log(pred_y) + (1-batch_gt_y) * np.log(1 - pred_y)
    loss = np.mean(-loss)
    return loss

def batch_gradient(batch_x, batch_gt_y, w_b, lr):
    pred_y = inference(w_b, batch_x)
    d = batch_x * (pred_y - batch_gt_y) * lr
    d = np.mean(d, axis=0).reshape(2,1)
    w_b -= d
    return w_b

def gen_sample_data(num):
    w_b = np.random.uniform(-2, 2, (2, 1))
    x = np.random.uniform(-2, 2, (num, 2))
    x[:, 1] = 1
    # y (num, 1)
    y = inference(w_b, x)
    y[y>0.5] = 1
    y[y<0.5] = 0
    return w_b, x, y

def train(x, gt_y, batch_size, lr, max_iter):
    w_b = np.zeros((2, 1))
    for i in range(max_iter):
        index = np.random.randint(x.shape[0], size = batch_size)
        batch_x = x[index]
        batch_gt_y = gt_y[index]
        pred_w_b = batch_gradient(batch_x, batch_gt_y, w_b, lr)
        loss = eval_loss(batch_x, batch_gt_y, w_b)
#         print('pred:',pred_w_b.flatten())
#         print('loss', loss)
    return pred_w_b, loss

def run():
    lr = 0.01
    num_samples = 100
    max_iter = 10000
    batch_size = 50
    w_b, x, y = gen_sample_data(num_samples)
    pred_w_b, loss = train(x, y, batch_size, lr, max_iter)
    print('w,b:',w_b.flatten())
    print('pred_w_b:',pred_w_b.flatten())
    print('loss', loss)

if __name__ == '__main__':
    run()
#     %timeit -o run()