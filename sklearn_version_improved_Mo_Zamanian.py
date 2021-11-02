# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% sklearn version: "LDA" algorithm using sklearn library



import numpy as np
from sklearn.discriminant_analysis import \
    LinearDiscriminantAnalysis

import matplotlib.pyplot as plt

# %% part 1 - load dataset and print summary

training_dataset = np.loadtxt('fld.txt', delimiter=',')
X_training_dataset = training_dataset[::, :2]
y_training_dataset = training_dataset[::, 2]

X_class_0_idx = np.where(y_training_dataset == 0)
X_class_1_idx = np.where(y_training_dataset == 1)
X_class_0 = X_training_dataset[X_class_0_idx]
X_class_1 = X_training_dataset[X_class_1_idx]

print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('Summary of the Dataset:')
print('Class "0" info: Shape - Head - Tail')
print('Shape:', X_class_0.shape)
print('Head:\n', X_class_0[:5, ::])
print('Tail:\n', X_class_0[-5:, ::])
print('----------------------------------------------------------------------')
print('Class "1" info: Shape - Head - Tail')
print('Shape:', X_class_1.shape)
print('Head:\n', X_class_1[:5, ::])
print('Tail:\n', X_class_1[-5:, ::])
print('----------------------------------------------------------------------')

# %% part 2 - perform Linear Discriminant Analysis (LDA)

LDA_object = LinearDiscriminantAnalysis(store_covariance=True)
LDA_object.fit(X_training_dataset, y_training_dataset)
Weight_vector = LDA_object.coef_[0]; intercept = LDA_object.intercept_
print('----------------------------------------------------------------------')
print('LDA >> Slope and Intercept:')
print('Slope (of log-posterior of LDA) =', Weight_vector,
      ',\nIntercept (of log-posterior of LDA) =', intercept)
print('----------------------------------------------------------------------')

pi_0 = len(X_class_0) / (len(X_class_1) + len(X_class_0))
pi_1 = len(X_class_1) / (len(X_class_1) + len(X_class_0))
Cst = np.log(pi_0/pi_1)
# Note >> covariance_array-like of shape (n_features, n_features)
# Note >> means_array-like of shape (n_classes, n_features)
w = np.dot(np.linalg.inv(LDA_object.covariance_),
           (LDA_object.means_[0] - LDA_object.means_[1]))
DC_term = Cst - 0.5 * np.dot((LDA_object.means_[0] + LDA_object.means_[1]).T, w)

# %% part 3 - prediction

predictions = (np.sign(np.dot(w, X_training_dataset.T) + DC_term) + 1) / 2
error_possibility_1 = sum(predictions != y_training_dataset)
error_possibility_2 = sum((1 - predictions) != y_training_dataset)
rel_error_1 = error_possibility_1 / len(y_training_dataset)
rel_error_2 = error_possibility_2 / len(y_training_dataset)

if rel_error_1 < rel_error_2:
    final_predictions = predictions
else:
    final_predictions = 1 - predictions

num_preds_to_print = 20
print(f'Some of predictions are: [first {num_preds_to_print}]\n',
      final_predictions[:num_preds_to_print])
print(f'Some of predictions are: [last {num_preds_to_print}]\n',
      final_predictions[-num_preds_to_print:])

# %% part 4 - error report

errorIndex = np.argwhere(final_predictions != y_training_dataset)
errorPts = X_training_dataset[errorIndex]
errorPts = np.squeeze(errorPts)

print('----------------------------------------------------------------------')
print('LDA >> Error:', 100 * min(rel_error_2, rel_error_1), '%.')
print('LDA >> LDA_object.score():',
      100 * LDA_object.score(X_training_dataset, y_training_dataset), '%.')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')

# %% part 5 - visualization

#  first plot
figure_width = 20
original_data_linewidth = 5
legend_fontsize = 20
plot_grid_option = False


plt.figure(figsize=(figure_width, figure_width / 1.618))
plt.scatter(X_class_0[:, 0],
            X_class_0[:, 1],
            c='r', marker='.',
            linewidths=original_data_linewidth)
plt.scatter(X_class_1[:, 0],
            X_class_1[:, 1],
            c='b', marker='.',
            linewidths=original_data_linewidth)

k0, k1 = 5, 3
plt.plot([- k0 * w[0], k1 * w[0]],
         [-k0 * w[1], k1 * w[1]],
         'g--', lw=5)

plt.xlabel('first axis (x0)', size=legend_fontsize)
plt.ylabel('second axis (x1)', size=legend_fontsize)
plt.legend(['LDA line', 'original data - class 0', 'original data - class 1'],
           fontsize=legend_fontsize)

plt.savefig('sklearn-improved-img-1.png', dpi=300)
plt.grid(True)
plt.savefig('sklearn-improved-img-1-grid.png', dpi=300)
plt.show()

#  second plot
plt.figure(figsize=(figure_width, figure_width / 1.618))
plt.scatter(X_class_0[:, 0],
            X_class_0[:, 1],
            c='r', marker='.',
            linewidths=original_data_linewidth)
plt.scatter(X_class_1[:, 0],
            X_class_1[:, 1],
            c='b', marker='.',
            linewidths=original_data_linewidth)

k0, k1 = 5, 3
plt.plot([- k0 * w[0], k1 * w[0]],
         [-k0 * w[1], k1 * w[1]],
         'g--', lw=5)


plt.scatter(errorPts[:, 0],
            errorPts[:, 1],
            c='orange',
            marker='o')

plt.xlabel('first axis (x0)', size=legend_fontsize)
plt.ylabel('second axis (x1)', size=legend_fontsize)
plt.legend(['LDA line',
            'original data - class 0',
            'original data - class 1',
            'LDA error samples'],
           fontsize=legend_fontsize)

plt.savefig('sklearn-improved-img-2.png', dpi=300)
plt.grid(True)
plt.savefig('sklearn-improved-img-2-grid.png', dpi=300)
plt.show()


