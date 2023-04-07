'''
Name: Pranath Reddy Kumbam
UFID: 8512-0977
NLP Project Codebase

Code to plot the confusion matrices for binary classification
'''

import matplotlib.pyplot as plt
import numpy as np

# Define the confusion matrix as a 2D numpy array
conf_matrix = np.array([[25284,   669],
                        [  700, 14014]])

# Define the classes (labels) of the confusion matrix
classes = ['Normal', 'Hate Speech']

# Define the title of the confusion matrix
title = 'Confusion Matrix for Distil BERT'

# Create a new figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the confusion matrix as a heatmap
im = ax.imshow(conf_matrix, cmap='Blues')

# Add a colorbar to the plot
cbar = ax.figure.colorbar(im, ax=ax)

# Set the tick labels for the x and y axes
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, fontsize=10)
ax.set_yticklabels(classes, fontsize=10)

# Loop over data dimensions and create text annotations
for i in range(len(classes)):
    for j in range(len(classes)):
        text = ax.text(j, i, conf_matrix[i, j],
                       ha="center", va="center", color="black", fontsize=12)

# Set the title of the plot
plt.title(title, fontsize=14)

# Add labels to the x and y axes
plt.xlabel('Predicted labels', fontsize=12)
plt.ylabel('True labels', fontsize=12)

# Adjust the margins of the plot to fit all labels and annotations
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

# Save the plot as a PNG file
plt.savefig('conf_mat_binary_DBERT.png', dpi=300)

# Show the plot
plt.show()


