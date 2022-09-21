import pandas as pd
import matplotlib.pyplot as plt
acc = pd.read_csv('result/acc3折.csv')
plt.plot(acc)
plt.title('ACC')
plt.ylabel('acc values')
plt.xlabel('epochs')
plt.show()

kappa = pd.read_csv('result/kappa3折.csv')
plt.plot(kappa)
plt.title('KAPPA')
plt.ylabel('kappa values')
plt.xlabel('epochs')
plt.show()

loss = pd.read_csv('result/loss3折.csv')
plt.plot(loss)
plt.title('LOSS')
plt.ylabel('loss values')
plt.xlabel('epochs')
# acc = pd.read_csv('result/acc3折.csv')
# plt.plot(acc)
plt.show()
