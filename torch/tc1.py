# C:\PY\torch\tc1.py
# Compute derivative
# Importing torch 
import torch 
  
# Initializing input tensors 
a = torch.tensor(15.0, requires_grad=True) 
b = torch.tensor(20.0, requires_grad=True) 
  
# Computing the output 
c = a * b 
  
# Computing the gradients 
c.backward() 
  
# Collecting the output gradient of the 
# output with respect to the input 'a' 
derivative_out_a = a.grad 
  
# Collecting the output gradient of the 
# output with respect to the input 'b' 
derivative_out_b = b.grad 
  
# Displaying the outputs 
print(f'c = {c}') 
print(f'Derivative of c with respect to a = {derivative_out_a}') 
print(f'Derivative of c with respect to b = {derivative_out_b}') 
