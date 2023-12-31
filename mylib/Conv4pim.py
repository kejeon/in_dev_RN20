# -*- coding: utf-8 -*-
"""Conv4PIM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dGP0mXRGYYeAkHVnWXX6X3j9VA9D1DPD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Conv2dSDK(torch.nn.Module):
  def __init__(self, kernel, pw_width, pw_height):
    super().__init__()

    if kernel.shape[2] != kernel.shape[3]:
      raise ValueError("Kernel is not square. Rectangular Kernel not supported.")

    if pw_height < kernel.shape[2] or pw_width < kernel.shape[3]:
      raise ValueError("Parallel window is smaller than the kernel.")

    # if pw_height == 3 and pw_width == 3:
      # print("WARNING: Parallel window size is 3. Use Conv2dIm2col instead.")

    self.kernel = kernel
    self.out_channels = kernel.shape[0]
    self.in_channels = kernel.shape[1]
    self.kernel_size = kernel.shape[2]

    self.pw_width = pw_width
    self.pw_height = pw_height

    self.weight_map = torch.nn.Parameter(self._gen_SDK_mapping(kernel))

  def _ordered_pairs_sum(self, x):
    a = torch.arange(x + 1)
    b = x - a
    pairs = torch.stack((a, b), dim=1)
    return pairs

  def _gen_SDK_mapping(self, my_tensor):
    h_diff = self.pw_height - self.kernel_size
    w_diff = self.pw_width - self.kernel_size

    ver_pads = self._ordered_pairs_sum(h_diff)
    hor_pads = self._ordered_pairs_sum(w_diff)

    SDK_mapping = []

    for i in range(len(ver_pads)):
      for j in range(len(hor_pads)):
        p2d = (hor_pads[j,0], hor_pads[j,1], ver_pads[i,0], ver_pads[i,1])
        padded_kernel =  F.pad(my_tensor, p2d, mode='constant', value=0)
        flat_kernel = padded_kernel.view(self.out_channels, -1)

        SDK_mapping.append(flat_kernel)

    SDK_mapping = torch.concat(SDK_mapping)

    return SDK_mapping

  def _forward(self, x):
    return F.linear(x, self.weight_map)

  def _slice_and_forward(self, x):
    num, depth, height, width = x.shape

    stride_ver = self.pw_height - self.kernel_size + 1
    stride_hor = self.pw_width  - self.kernel_size + 1

    pad_ver = (height + 2 - self.pw_height) % stride_ver
    pad_hor = (width  + 2 - self.pw_width)  % stride_hor

    slide_ver = math.ceil((height + 2 - self.pw_height) / stride_ver) + 1
    slide_hor = math.ceil((width  + 2 - self.pw_width ) / stride_hor) + 1

    padded_x = F.pad(x, (1, 1 + pad_hor, 1, 1 + pad_ver), 
                     mode='constant', value=0)

    flat_windows = F.unfold(padded_x, 
                            kernel_size=(self.pw_height, self.pw_width), 
                            stride=(stride_ver, stride_hor)).transpose(1,2)

    lin_out = self._forward(flat_windows)
    # lin_out = F.linear(flat_windows, self.weight_map)
    # print(lin_out.shape)

    lin_out = lin_out.reshape(num, slide_ver, slide_hor, 
                              self.pw_height - self.kernel_size + 1, 
                              self.pw_width  - self.kernel_size + 1, self.out_channels)
    # print(lin_out.shape)

    lin_out = lin_out.transpose(2,3)
    lin_out = lin_out.reshape(num, 
                              height+int(pad_ver/2), 
                              width+int(pad_hor/2), 
                              self.out_channels)
    lin_out = lin_out.transpose(3,1).transpose(3,2)
    # print(lin_out.shape)
    lin_out = lin_out[:,:,:height,:width]
    return lin_out

  def forward(self, input):
    return self._slice_and_forward(input)

  def string(self):
    return 'testing'

class Conv2dSDK_QR(Conv2dSDK):
  def __init__(self, kernel, grad, pw_width, pw_height, rank, svd_mode='vanilla', alpha = 0.5):
    super().__init__(kernel, pw_width, pw_height)
    self.rank = rank
    self.original_weight_map = torch.tensor(self.weight_map)
    Q, R = self._SVD(grad, svd_mode)
    self.Q = torch.nn.Parameter(torch.tensor(Q))
    self.R = torch.nn.Parameter(torch.tensor(R))

  def _forward(self, x):
    # print((self.Q).shape)
    # print((self.R).shape)
    # print((self.Q @ self.R).shape)
    # print(x.shape)
    return F.linear(x, self.Q @ self.R)

  def _SVD(self, grad, svd_mode):
    weighted_map = self.original_weight_map.cpu().detach().numpy()
    if grad is not None:
      SDK_grad = self._gen_SDK_mapping(grad)
      SDK_grad_np = grad.cpu().detach().numpy()
      # pre_weight = np.diag(np.sum(SDK_grad_np, axis=0))
      # post_weight = np.diag(np.sum(SDK_grad_np, axis=1))
      pre_weight = np.diag(np.diag(SDK_grad_np.T @ SDK_grad_np))
      post_weight = np.diag(np.diag(SDK_grad_np @ SDK_grad_np.T))
      if svd_mode is 'vanilla':
        weighted_map = weighted_map
      elif svd_mode is 'fisher':
        weighted_map = pre_weight @ weighted_map
      elif svd_mode is 'jeon':
        weighted_map = pre_weight @ weighted_map @ post_weight
      elif svd_mode is 'jeon_post':
        weighted_map = weighted_map @ post_weight

    u, s, vh = np.linalg.svd(weighted_map, 
                             full_matrices=False)
    u_t = u[:,0:self.rank]
    s_t = np.diag(s[:self.rank])
    v_t = vh[:self.rank,:]
    # print(u_t.shape)
    # print(s_t.shape)
    # print(v_t.shape)
    Q = u_t@s_t
    R = v_t

    if grad is not None:
      if svd_mode is 'vanilla':
        return Q, R
      elif svd_mode is 'fisher':
        Q = np.linalg.inv(pre_weight) @ Q
      elif svd_mode is 'jeon':
        Q = np.linalg.inv(pre_weight) @ Q
        R = R @ np.linalg.inv(post_weight)
      elif svd_mode is 'jeon_post':
        R = R @ np.linalg.inv(post_weight)
    # print(Q.shape)
    # print(R.shape)
    return Q, R

def test_script():
  # gen random data
  img_num = 1
  img_width = 8
  input_channel = 32
  kernel_size = 3
  output_channel = 64

  # create a 1D tensor with values ranging from 0 to 8*8*64-1
  # img = torch.arange(img_num*img_width*img_width*input_channel)
  # img = img.reshape(img_num, input_channel, img_width, img_width)
  img = torch.randn(img_num, input_channel, img_width, img_width)

  # create a 4D random tensor
  kernel = torch.randn(output_channel, input_channel, kernel_size, kernel_size)
  # kernel2 = torch.randn(output_channel, input_channel, kernel_size, kernel_size)

  my_conv1 = Conv2dSDK_QR(rank=40, kernel=kernel, pw_width=3, pw_height=3)
  lin_out = my_conv1(img)
  output = F.conv2d(img, kernel, padding=1)

  # See that the two operation is identical
  l1_norm = torch.norm(lin_out - output, p=1)
  print(l1_norm)

