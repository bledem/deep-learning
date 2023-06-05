"""
Stack 1x1, 3x3, 5x5 filters maps (and pooling) altogether and let the model 
learns the best combination among these ones.
1x1 conv solves computational costs of convlayers. It acts as a linear projection, and enables to reduces the dimension of 
the previous activation. 
"""
