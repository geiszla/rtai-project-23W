import torch

x = torch.tensor([[-1.5788e-02, -0.0000e+00, -0.0000e+00,  1.2393e-02,  3.7474e-02,
          0.0000e+00,  2.3180e-02, -1.0438e-03,  5.8113e-02, -0.0000e+00,
         -8.5743e-03, -0.0000e+00, -9.7296e-06, -2.4898e-02,  2.2644e-02,
         -0.0000e+00,  0.0000e+00, -4.2123e-03, -6.1241e-03, -3.5000e-04],
        [-3.7856e-02, -0.0000e+00,  0.0000e+00, -4.2656e-03,  3.8671e-03,
          0.0000e+00, -1.2798e-02,  1.5257e-02,  5.2019e-02, -0.0000e+00,
          1.0474e-01, -0.0000e+00,  8.1378e-05,  3.5885e-02,  2.8939e-03,
         -0.0000e+00,  0.0000e+00,  4.5919e-03,  4.5168e-02,  1.0669e-02],
        [ 6.3353e-03, -0.0000e+00, -0.0000e+00,  2.6482e-02,  1.0602e-02,
          0.0000e+00,  1.4534e-02, -3.6253e-02,  2.0586e-02,  0.0000e+00,
         -9.9203e-03, -0.0000e+00,  3.6449e-05,  1.0259e-02, -1.0805e-04,
          0.0000e+00, -0.0000e+00,  2.3251e-02, -5.0327e-02, -1.4418e-01],
        [ 7.3107e-03, -0.0000e+00, -0.0000e+00,  2.0662e-02,  4.6385e-03,
         -0.0000e+00, -6.4052e-02,  2.0586e-02,  1.2181e-02,  0.0000e+00,
          5.4441e-03,  0.0000e+00,  1.2038e-04,  2.8718e-02, -1.5310e-02,
         -0.0000e+00, -0.0000e+00,  4.1153e-02, -3.6727e-02,  5.4647e-02],
        [ 2.1895e-02, -0.0000e+00, -0.0000e+00,  2.4933e-02, -3.7161e-02,
          0.0000e+00, -5.0689e-02,  3.6398e-02, -2.6010e-02, -0.0000e+00,
          2.1849e-02, -0.0000e+00,  8.2472e-05,  4.8508e-02, -4.0064e-02,
          0.0000e+00, -0.0000e+00,  3.7747e-02, -4.0376e-02, -2.8414e-04],
        [ 6.8767e-02, -0.0000e+00,  0.0000e+00,  1.7814e-02, -5.4398e-02,
         -0.0000e+00, -2.3918e-02,  6.1336e-02, -2.5382e-02, -0.0000e+00,
         -1.4470e-03, -0.0000e+00,  4.7243e-05,  4.1256e-02, -1.2421e-02,
          0.0000e+00, -0.0000e+00,  2.0681e-02, -1.1243e-02, -3.5391e-02],
        [ 3.0130e-02,  0.0000e+00, -0.0000e+00,  3.6003e-02,  2.0394e-02,
         -0.0000e+00,  9.0814e-02,  3.4284e-02,  9.2707e-02,  0.0000e+00,
         -1.0535e-02,  0.0000e+00, -6.7466e-05, -6.6340e-02, -1.5181e-02,
         -0.0000e+00,  0.0000e+00,  2.2607e-02,  3.4393e-02, -1.5353e-02],
        [ 1.5542e-02,  0.0000e+00,  0.0000e+00, -4.6494e-02, -6.9311e-03,
          0.0000e+00,  6.3130e-02, -3.3884e-02,  3.2807e-02, -0.0000e+00,
         -9.9759e-02,  0.0000e+00, -1.3065e-04, -5.3905e-02, -7.9240e-03,
          0.0000e+00,  0.0000e+00, -4.8426e-02,  3.8615e-02, -1.0430e-01],
        [-2.7520e-02, -0.0000e+00,  0.0000e+00, -4.1113e-02, -1.7440e-02,
         -0.0000e+00,  6.0401e-02,  2.9606e-02, -6.0931e-02, -0.0000e+00,
          1.0635e-02,  0.0000e+00,  7.5753e-05,  2.7353e-02,  1.2022e-01,
         -0.0000e+00,  0.0000e+00, -8.3083e-02,  5.0711e-02, -5.3102e-02],
        [-2.0626e-03,  0.0000e+00, -0.0000e+00,  3.0320e-02, -1.1184e-02,
          0.0000e+00,  9.3826e-02,  9.5405e-03, -3.1383e-03,  0.0000e+00,
          2.8367e-02, -0.0000e+00, -6.7360e-06, -1.0923e-02, -3.7649e-02,
         -0.0000e+00,  0.0000e+00,  4.6823e-02,  3.4320e-02, -1.3794e-02],
        [-4.6307e-04, -0.0000e+00, -0.0000e+00,  3.8968e-02,  1.4370e-02,
          0.0000e+00, -2.5847e-02, -5.9292e-02, -2.3809e-02,  0.0000e+00,
         -6.2397e-03, -0.0000e+00, -4.5476e-06,  2.9576e-02, -2.8299e-02,
          0.0000e+00, -0.0000e+00,  3.6940e-04, -6.9264e-02, -5.8862e-02],
        [-2.3879e-02, -0.0000e+00,  0.0000e+00, -8.8744e-03, -8.9392e-03,
         -0.0000e+00,  7.2653e-04,  4.4534e-02,  4.3020e-02,  0.0000e+00,
          4.1390e-02, -0.0000e+00,  1.0368e-04,  1.5077e-02,  5.9637e-02,
         -0.0000e+00,  0.0000e+00, -2.7916e-02, -6.5337e-02, -2.8226e-02],
        [ 2.5738e-02, -0.0000e+00, -0.0000e+00,  2.0298e-02, -3.8016e-02,
          0.0000e+00,  1.6260e-02, -3.6416e-02, -4.7993e-02, -0.0000e+00,
          5.2835e-03,  0.0000e+00, -3.2005e-05, -2.4790e-03, -4.8266e-02,
          0.0000e+00, -0.0000e+00,  2.1845e-02,  1.4013e-02,  2.5379e-02],
        [ 4.2539e-02, -0.0000e+00, -0.0000e+00,  6.7268e-02,  3.4850e-02,
          0.0000e+00,  2.4924e-02, -4.4253e-02,  1.1660e-02, -0.0000e+00,
         -8.5866e-02,  0.0000e+00, -1.5563e-05, -4.9011e-02, -3.7423e-02,
         -0.0000e+00, -0.0000e+00,  6.2259e-02,  9.8944e-03, -1.2244e-02],
        [ 2.5956e-02, -0.0000e+00,  0.0000e+00,  2.1106e-03,  1.6375e-02,
         -0.0000e+00, -2.7442e-02,  2.6478e-02, -4.1217e-02,  0.0000e+00,
          6.3915e-03,  0.0000e+00,  7.3306e-05,  1.6664e-02,  1.4446e-02,
         -0.0000e+00, -0.0000e+00,  7.7879e-02,  5.7034e-02,  1.1270e-01]])

y = torch.tensor([[-1.5788e-02,  0.0000e+00,  0.0000e+00,  1.2393e-02,  3.7474e-02,
          0.0000e+00,  2.3180e-02, -1.0438e-03,  5.8113e-02,  0.0000e+00,
         -8.5743e-03,  0.0000e+00, -9.7301e-06, -2.4898e-02,  2.2644e-02,
          0.0000e+00,  0.0000e+00, -4.2123e-03, -6.1241e-03, -3.5001e-04],
        [-3.7856e-02,  0.0000e+00,  0.0000e+00, -4.2656e-03,  3.8671e-03,
          0.0000e+00, -1.2798e-02,  1.5257e-02,  5.2019e-02,  0.0000e+00,
          1.0474e-01,  0.0000e+00,  8.1381e-05,  3.5885e-02,  2.8939e-03,
          0.0000e+00,  0.0000e+00,  4.5919e-03,  4.5168e-02,  1.0669e-02],
        [ 6.3353e-03,  0.0000e+00,  0.0000e+00,  2.6482e-02,  1.0602e-02,
          0.0000e+00,  1.4534e-02, -3.6253e-02,  2.0586e-02,  0.0000e+00,
         -9.9202e-03,  0.0000e+00,  3.6451e-05,  1.0259e-02, -1.0804e-04,
          0.0000e+00,  0.0000e+00,  2.3251e-02, -5.0327e-02, -1.4418e-01],
        [ 7.3107e-03,  0.0000e+00,  0.0000e+00,  2.0662e-02,  4.6385e-03,
          0.0000e+00, -6.4052e-02,  2.0586e-02,  1.2181e-02,  0.0000e+00,
          5.4441e-03,  0.0000e+00,  1.2039e-04,  2.8718e-02, -1.5310e-02,
          0.0000e+00,  0.0000e+00,  4.1153e-02, -3.6727e-02,  5.4647e-02],
        [ 2.1895e-02,  0.0000e+00,  0.0000e+00,  2.4933e-02, -3.7161e-02,
          0.0000e+00, -5.0689e-02,  3.6398e-02, -2.6010e-02,  0.0000e+00,
          2.1849e-02,  0.0000e+00,  8.2476e-05,  4.8508e-02, -4.0064e-02,
          0.0000e+00,  0.0000e+00,  3.7747e-02, -4.0376e-02, -2.8413e-04],
        [ 6.8767e-02,  0.0000e+00,  0.0000e+00,  1.7814e-02, -5.4398e-02,
          0.0000e+00, -2.3918e-02,  6.1336e-02, -2.5382e-02,  0.0000e+00,
         -1.4470e-03,  0.0000e+00,  4.7245e-05,  4.1256e-02, -1.2421e-02,
          0.0000e+00,  0.0000e+00,  2.0681e-02, -1.1243e-02, -3.5391e-02],
        [ 3.0130e-02,  0.0000e+00,  0.0000e+00,  3.6003e-02,  2.0394e-02,
          0.0000e+00,  9.0814e-02,  3.4284e-02,  9.2707e-02,  0.0000e+00,
         -1.0535e-02,  0.0000e+00, -6.7469e-05, -6.6340e-02, -1.5181e-02,
          0.0000e+00,  0.0000e+00,  2.2607e-02,  3.4393e-02, -1.5353e-02],
        [ 1.5542e-02,  0.0000e+00,  0.0000e+00, -4.6494e-02, -6.9311e-03,
          0.0000e+00,  6.3130e-02, -3.3884e-02,  3.2807e-02,  0.0000e+00,
         -9.9759e-02,  0.0000e+00, -1.3066e-04, -5.3905e-02, -7.9240e-03,
          0.0000e+00,  0.0000e+00, -4.8426e-02,  3.8615e-02, -1.0430e-01],
        [-2.7520e-02,  0.0000e+00,  0.0000e+00, -4.1113e-02, -1.7440e-02,
          0.0000e+00,  6.0401e-02,  2.9606e-02, -6.0931e-02,  0.0000e+00,
          1.0635e-02,  0.0000e+00,  7.5757e-05,  2.7353e-02,  1.2022e-01,
          0.0000e+00,  0.0000e+00, -8.3083e-02,  5.0711e-02, -5.3102e-02],
        [-2.0626e-03,  0.0000e+00,  0.0000e+00,  3.0320e-02, -1.1184e-02,
          0.0000e+00,  9.3826e-02,  9.5405e-03, -3.1383e-03,  0.0000e+00,
          2.8367e-02,  0.0000e+00, -6.7363e-06, -1.0923e-02, -3.7649e-02,
          0.0000e+00,  0.0000e+00,  4.6823e-02,  3.4320e-02, -1.3794e-02],
        [-4.6307e-04,  0.0000e+00,  0.0000e+00,  3.8968e-02,  1.4370e-02,
          0.0000e+00, -2.5847e-02, -5.9292e-02, -2.3809e-02,  0.0000e+00,
         -6.2397e-03,  0.0000e+00, -4.5478e-06,  2.9576e-02, -2.8299e-02,
          0.0000e+00,  0.0000e+00,  3.6939e-04, -6.9264e-02, -5.8862e-02],
        [-2.3879e-02,  0.0000e+00,  0.0000e+00, -8.8744e-03, -8.9392e-03,
          0.0000e+00,  7.2653e-04,  4.4534e-02,  4.3020e-02,  0.0000e+00,
          4.1390e-02,  0.0000e+00,  1.0368e-04,  1.5077e-02,  5.9637e-02,
          0.0000e+00,  0.0000e+00, -2.7916e-02, -6.5337e-02, -2.8226e-02],
        [ 2.5738e-02,  0.0000e+00,  0.0000e+00,  2.0298e-02, -3.8016e-02,
          0.0000e+00,  1.6260e-02, -3.6416e-02, -4.7993e-02,  0.0000e+00,
          5.2835e-03,  0.0000e+00, -3.2006e-05, -2.4790e-03, -4.8266e-02,
          0.0000e+00,  0.0000e+00,  2.1845e-02,  1.4013e-02,  2.5379e-02],
        [ 4.2539e-02,  0.0000e+00,  0.0000e+00,  6.7268e-02,  3.4850e-02,
          0.0000e+00,  2.4924e-02, -4.4253e-02,  1.1660e-02,  0.0000e+00,
         -8.5866e-02,  0.0000e+00, -1.5564e-05, -4.9011e-02, -3.7423e-02,
          0.0000e+00,  0.0000e+00,  6.2259e-02,  9.8944e-03, -1.2244e-02],
        [ 2.5956e-02,  0.0000e+00,  0.0000e+00,  2.1106e-03,  1.6375e-02,
          0.0000e+00, -2.7442e-02,  2.6478e-02, -4.1217e-02,  0.0000e+00,
          6.3915e-03,  0.0000e+00,  7.3310e-05,  1.6664e-02,  1.4446e-02,
          0.0000e+00,  0.0000e+00,  7.7879e-02,  5.7034e-02,  1.1270e-01]])

# tensor1 = x
# tensor2 = y
# # Find where the tensors are not equal
# not_equal = tensor1 != tensor2

# # Extract the unequal values from each tensor
# unequal_values_tensor1 = tensor1[not_equal]
# unequal_values_tensor2 = tensor2[not_equal]

# # Combine the values side by side
# unequal_pairs = torch.stack((unequal_values_tensor1, unequal_values_tensor2), dim=1)

# print(unequal_pairs)



def print_unequal_elements(tensor1, tensor2):
    # Compare tensors element-wise 
    not_equal_tensor = torch.eq(tensor1, tensor2) 
    # Find indices where elements are not equal 
    unequal_indices = torch.nonzero(~not_equal_tensor)
    # Print indices and values of unequal elements
    for index in unequal_indices: 
        idx_tuple = tuple(index.tolist())
        value_tensor1 = tensor1[idx_tuple] 
        value_tensor2 = tensor2[idx_tuple] 
        print(f"Index {idx_tuple}: Tensor1[{idx_tuple}] = {value_tensor1}, Tensor2[{idx_tuple}] = {value_tensor2}, Difference = {value_tensor1 - value_tensor2}")

print_unequal_elements(x, y)