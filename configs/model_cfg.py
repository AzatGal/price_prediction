from easydict import EasyDict

from configs.data_cfg import cfg as data_cfg

cfg = EasyDict()

cfg.num_heads = 4
cfg.embed_dim = 16 * cfg.num_heads  # 24
cfg.num_blocks = 2  # 3
cfg.act = 'SiLU'  # SiLU
cfg.num_embed_features = (data_cfg.data_transformer.num_bins +
                          data_cfg.data_transformer.num_cats)
# cfg.pred_dim = 1  # cfg.num_embed_features[0]  # 1  #
cfg.attn_dropout = 0.0
cfg.mlp_dropout = 0.1
cfg.dropout = 0.1
cfg.compression_factor = 0.75
cfg.compression = 'Head'  # Head KV Layer
cfg.mlp_dim_factor = 1  # 5 / 3
cfg.attn = 'Attn'  # Linear
cfg.mlp = 'GLUMLP'
cfg.norm = 'LayerNorm'
cfg.log_softmax = False  # False True

# print(sum(cfg.num_embed_features))

"""
cpu
Epoch 1/125
train loss: 0.7867 - metric: 0.514 - time: 5.9 (5.9) 
valid loss: 0.8013 - metric: 0.440 - time: 5.9 (0.2) 
best
Epoch 2/125
train loss: 0.5403 - metric: 0.311 - time: 10.6 (4.7) 
valid loss: 0.3789 - metric: 0.164 - time: 10.6 (0.2) 
best
Epoch 3/125
train loss: 0.2983 - metric: 0.160 - time: 15.3 (4.8) 
valid loss: 0.3562 - metric: 0.126 - time: 15.3 (0.2) 
best
Epoch 4/125
train loss: 0.2453 - metric: 0.133 - time: 19.9 (4.6) 
valid loss: 0.2879 - metric: 0.129 - time: 19.9 (0.2) 
Epoch 5/125
train loss: 0.2244 - metric: 0.123 - time: 24.7 (4.8) 
valid loss: 0.3353 - metric: 0.109 - time: 24.7 (0.2) 
best
Epoch 6/125
train loss: 0.2087 - metric: 0.113 - time: 29.7 (5.0) 
valid loss: 0.3043 - metric: 0.114 - time: 29.7 (0.2) 
Epoch 7/125
train loss: 0.1961 - metric: 0.107 - time: 34.3 (4.6) 
valid loss: 0.2907 - metric: 0.092 - time: 34.3 (0.2) 
best
Epoch 8/125
train loss: 0.1828 - metric: 0.099 - time: 38.9 (4.6) 
valid loss: 0.2615 - metric: 0.086 - time: 38.9 (0.4) 
best
Epoch 9/125
train loss: 0.1674 - metric: 0.091 - time: 43.5 (4.6) 
valid loss: 0.2882 - metric: 0.084 - time: 43.5 (0.2) 
best
Epoch 10/125
train loss: 0.1540 - metric: 0.084 - time: 48.1 (4.6) 
valid loss: 0.2410 - metric: 0.076 - time: 48.1 (0.2) 
best
Epoch 11/125
train loss: 0.1546 - metric: 0.084 - time: 52.8 (4.7) 
valid loss: 0.2491 - metric: 0.080 - time: 52.8 (0.2) 
Epoch 12/125
train loss: 0.1404 - metric: 0.076 - time: 57.3 (4.6) 
valid loss: 0.2383 - metric: 0.077 - time: 57.3 (0.2) 
Epoch 13/125
train loss: 0.1343 - metric: 0.073 - time: 61.9 (4.6) 
valid loss: 0.2251 - metric: 0.068 - time: 61.9 (0.2) 
best
Epoch 14/125
train loss: 0.1334 - metric: 0.073 - time: 66.7 (4.8) 
valid loss: 0.2415 - metric: 0.071 - time: 66.7 (0.2) 
Epoch 15/125
train loss: 0.1297 - metric: 0.071 - time: 71.3 (4.6) 
valid loss: 0.2365 - metric: 0.075 - time: 71.3 (0.2) 
Epoch 16/125
train loss: 0.1280 - metric: 0.070 - time: 76.1 (4.7) 
valid loss: 0.2089 - metric: 0.066 - time: 76.1 (0.2) 
best
Epoch 17/125
train loss: 0.1269 - metric: 0.069 - time: 80.8 (4.8) 
valid loss: 0.2340 - metric: 0.073 - time: 80.8 (0.3) 
Epoch 18/125
train loss: 0.1215 - metric: 0.066 - time: 85.4 (4.6) 
valid loss: 0.2235 - metric: 0.069 - time: 85.4 (0.2) 
Epoch 19/125
train loss: 0.1179 - metric: 0.064 - time: 89.9 (4.5) 
valid loss: 0.2241 - metric: 0.068 - time: 89.9 (0.2) 
Epoch 20/125
train loss: 0.1172 - metric: 0.064 - time: 94.7 (4.7) 
valid loss: 0.2305 - metric: 0.073 - time: 94.7 (0.2) 
Epoch 21/125
train loss: 0.1146 - metric: 0.063 - time: 99.4 (4.8) 
valid loss: 0.2469 - metric: 0.074 - time: 99.4 (0.3) 
Epoch 22/125
train loss: 0.1139 - metric: 0.062 - time: 104.3 (4.8) 
valid loss: 0.2204 - metric: 0.063 - time: 104.3 (0.3) 
best
Epoch 23/125
train loss: 0.1099 - metric: 0.060 - time: 110.4 (6.1) 
valid loss: 0.2217 - metric: 0.068 - time: 110.4 (0.3) 
Epoch 24/125
train loss: 0.1100 - metric: 0.060 - time: 115.2 (4.8) 
valid loss: 0.1971 - metric: 0.063 - time: 115.2 (0.2) 
best
Epoch 25/125
train loss: 0.1095 - metric: 0.060 - time: 120.0 (4.7) 
valid loss: 0.2053 - metric: 0.062 - time: 120.0 (0.2) 
best
Epoch 26/125
train loss: 0.1122 - metric: 0.061 - time: 124.8 (4.8) 
valid loss: 0.2140 - metric: 0.060 - time: 124.8 (0.2) 
best
Epoch 27/125
train loss: 0.1052 - metric: 0.057 - time: 129.8 (5.0) 
valid loss: 0.2202 - metric: 0.064 - time: 129.8 (0.4) 
Epoch 28/125
train loss: 0.1038 - metric: 0.057 - time: 135.5 (5.7) 
valid loss: 0.2129 - metric: 0.062 - time: 135.5 (0.2) 
Epoch 29/125
train loss: 0.1043 - metric: 0.057 - time: 140.5 (5.0) 
valid loss: 0.2069 - metric: 0.058 - time: 140.5 (0.3) 
best
Epoch 30/125
train loss: 0.1030 - metric: 0.056 - time: 145.1 (4.7) 
valid loss: 0.1961 - metric: 0.061 - time: 145.1 (0.2) 
Epoch 31/125
train loss: 0.1013 - metric: 0.055 - time: 149.9 (4.8) 
valid loss: 0.1895 - metric: 0.062 - time: 149.9 (0.4) 
Epoch 32/125
train loss: 0.1003 - metric: 0.055 - time: 154.7 (4.8) 
valid loss: 0.1903 - metric: 0.060 - time: 154.7 (0.2) 
Epoch 33/125
train loss: 0.1000 - metric: 0.055 - time: 159.3 (4.6) 
valid loss: 0.1984 - metric: 0.060 - time: 159.3 (0.2) 
Epoch 34/125
train loss: 0.0978 - metric: 0.053 - time: 164.8 (5.5) 
valid loss: 0.2021 - metric: 0.060 - time: 164.8 (0.4) 
Epoch 35/125
train loss: 0.0977 - metric: 0.053 - time: 169.9 (5.1) 
valid loss: 0.1908 - metric: 0.057 - time: 169.9 (0.3) 
best
Epoch 36/125
train loss: 0.0968 - metric: 0.053 - time: 174.8 (4.8) 
valid loss: 0.1972 - metric: 0.059 - time: 174.8 (0.3) 
Epoch 37/125
train loss: 0.0953 - metric: 0.052 - time: 179.5 (4.8) 
valid loss: 0.1977 - metric: 0.058 - time: 179.5 (0.2) 
Epoch 38/125
train loss: 0.0947 - metric: 0.052 - time: 184.4 (4.8) 
valid loss: 0.1986 - metric: 0.060 - time: 184.4 (0.2) 
Epoch 39/125
train loss: 0.0937 - metric: 0.051 - time: 189.0 (4.7) 
valid loss: 0.1994 - metric: 0.059 - time: 189.0 (0.3) 
Epoch 40/125
train loss: 0.0940 - metric: 0.051 - time: 193.8 (4.8) 
valid loss: 0.1940 - metric: 0.058 - time: 193.8 (0.3) 
Epoch 41/125
train loss: 0.0931 - metric: 0.051 - time: 198.6 (4.7) 
valid loss: 0.1975 - metric: 0.059 - time: 198.6 (0.3) 
Epoch 42/125
train loss: 0.0931 - metric: 0.051 - time: 203.4 (4.8) 
valid loss: 0.2252 - metric: 0.064 - time: 203.4 (0.2) 
Epoch 43/125
train loss: 0.0926 - metric: 0.050 - time: 208.1 (4.7) 
valid loss: 0.1991 - metric: 0.057 - time: 208.1 (0.3) 
Epoch 44/125
train loss: 0.0927 - metric: 0.051 - time: 212.9 (4.8) 
valid loss: 0.2204 - metric: 0.061 - time: 212.9 (0.2) 
Epoch 45/125
train loss: 0.0900 - metric: 0.049 - time: 217.6 (4.6) 
valid loss: 0.2200 - metric: 0.060 - time: 217.6 (0.3) 
Epoch 46/125
train loss: 0.0903 - metric: 0.049 - time: 222.4 (4.8) 
valid loss: 0.2025 - metric: 0.058 - time: 222.4 (0.3) 
Epoch 47/125
train loss: 0.0892 - metric: 0.049 - time: 227.0 (4.6) 
valid loss: 0.2231 - metric: 0.059 - time: 227.0 (0.2) 
Epoch 48/125
train loss: 0.0885 - metric: 0.048 - time: 231.6 (4.6) 
valid loss: 0.2229 - metric: 0.062 - time: 231.6 (0.2) 
Epoch 49/125
train loss: 0.0891 - metric: 0.049 - time: 236.4 (4.8) 
valid loss: 0.2045 - metric: 0.056 - time: 236.4 (0.2) 
best
Epoch 50/125
train loss: 0.0883 - metric: 0.048 - time: 241.2 (4.8) 
valid loss: 0.2021 - metric: 0.058 - time: 241.2 (0.2) 
Epoch 51/125
train loss: 0.0870 - metric: 0.047 - time: 245.9 (4.7) 
valid loss: 0.2118 - metric: 0.061 - time: 245.9 (0.3) 
Epoch 52/125
train loss: 0.0871 - metric: 0.047 - time: 250.8 (4.9) 
valid loss: 0.1859 - metric: 0.057 - time: 250.8 (0.2) 
Epoch 53/125
train loss: 0.0863 - metric: 0.047 - time: 255.7 (5.0) 
valid loss: 0.2101 - metric: 0.060 - time: 255.7 (0.2) 
Epoch 54/125
train loss: 0.0855 - metric: 0.047 - time: 260.4 (4.7) 
valid loss: 0.2023 - metric: 0.059 - time: 260.4 (0.2) 
Epoch 55/125
train loss: 0.0864 - metric: 0.047 - time: 265.3 (4.9) 
valid loss: 0.2017 - metric: 0.058 - time: 265.3 (0.3) 
Epoch 56/125
train loss: 0.0854 - metric: 0.046 - time: 270.3 (5.0) 
valid loss: 0.2093 - metric: 0.059 - time: 270.3 (0.3) 
Epoch 57/125
train loss: 0.0851 - metric: 0.046 - time: 275.2 (4.9) 
valid loss: 0.1871 - metric: 0.055 - time: 275.2 (0.2) 
best
Epoch 58/125
train loss: 0.0843 - metric: 0.046 - time: 280.1 (4.9) 
valid loss: 0.1958 - metric: 0.056 - time: 280.1 (0.3) 
Epoch 59/125
train loss: 0.0844 - metric: 0.046 - time: 284.9 (4.8) 
valid loss: 0.1987 - metric: 0.056 - time: 284.9 (0.2) 
Epoch 60/125
train loss: 0.0830 - metric: 0.045 - time: 289.7 (4.8) 
valid loss: 0.2041 - metric: 0.056 - time: 289.7 (0.3) 
Epoch 61/125
train loss: 0.0841 - metric: 0.046 - time: 294.4 (4.7) 
valid loss: 0.2024 - metric: 0.055 - time: 294.4 (0.2) 
Epoch 62/125
train loss: 0.0833 - metric: 0.045 - time: 299.0 (4.7) 
valid loss: 0.1999 - metric: 0.055 - time: 299.0 (0.2) 
Epoch 63/125
train loss: 0.0831 - metric: 0.045 - time: 303.8 (4.7) 
valid loss: 0.1949 - metric: 0.055 - time: 303.8 (0.2) 
best
Epoch 64/125
train loss: 0.0816 - metric: 0.044 - time: 308.5 (4.7) 
valid loss: 0.2092 - metric: 0.058 - time: 308.5 (0.2) 
Epoch 65/125
train loss: 0.0817 - metric: 0.044 - time: 313.1 (4.7) 
valid loss: 0.2012 - metric: 0.053 - time: 313.1 (0.2) 
best
Epoch 66/125
train loss: 0.0821 - metric: 0.045 - time: 317.8 (4.7) 
valid loss: 0.2010 - metric: 0.053 - time: 317.8 (0.2) 
Epoch 67/125
train loss: 0.0817 - metric: 0.044 - time: 322.5 (4.7) 
valid loss: 0.2140 - metric: 0.058 - time: 322.5 (0.3) 
Epoch 68/125
train loss: 0.0811 - metric: 0.044 - time: 327.2 (4.6) 
valid loss: 0.2093 - metric: 0.058 - time: 327.2 (0.3) 
Epoch 69/125
train loss: 0.0797 - metric: 0.043 - time: 331.9 (4.7) 
valid loss: 0.2088 - metric: 0.057 - time: 331.9 (0.2) 
Epoch 70/125
train loss: 0.0804 - metric: 0.044 - time: 336.6 (4.7) 
valid loss: 0.2029 - metric: 0.056 - time: 336.6 (0.3) 
Epoch 71/125
train loss: 0.0806 - metric: 0.044 - time: 341.2 (4.7) 
valid loss: 0.2052 - metric: 0.056 - time: 341.2 (0.2) 
Epoch 72/125
train loss: 0.0795 - metric: 0.043 - time: 346.0 (4.8) 
valid loss: 0.2098 - metric: 0.054 - time: 346.0 (0.2) 
Epoch 73/125
train loss: 0.0793 - metric: 0.043 - time: 350.8 (4.8) 
valid loss: 0.2079 - metric: 0.056 - time: 350.8 (0.3) 
Epoch 74/125
train loss: 0.0790 - metric: 0.043 - time: 355.5 (4.7) 
valid loss: 0.1903 - metric: 0.055 - time: 355.5 (0.3) 
Epoch 75/125
train loss: 0.0792 - metric: 0.043 - time: 360.2 (4.7) 
valid loss: 0.1899 - metric: 0.055 - time: 360.2 (0.2) 
Epoch 76/125
train loss: 0.0793 - metric: 0.043 - time: 364.9 (4.7) 
valid loss: 0.2111 - metric: 0.053 - time: 364.9 (0.3) 
best
Epoch 77/125
train loss: 0.0791 - metric: 0.043 - time: 369.6 (4.7) 
valid loss: 0.1961 - metric: 0.055 - time: 369.6 (0.3) 
Epoch 78/125
train loss: 0.0782 - metric: 0.042 - time: 374.3 (4.7) 
valid loss: 0.1996 - metric: 0.056 - time: 374.3 (0.2) 
Epoch 79/125
train loss: 0.0785 - metric: 0.043 - time: 378.9 (4.6) 
valid loss: 0.2009 - metric: 0.054 - time: 378.9 (0.3) 
Epoch 80/125
train loss: 0.0779 - metric: 0.042 - time: 383.7 (4.8) 
valid loss: 0.2033 - metric: 0.056 - time: 383.7 (0.2) 
Epoch 81/125
train loss: 0.0784 - metric: 0.042 - time: 388.3 (4.7) 
valid loss: 0.2104 - metric: 0.059 - time: 388.3 (0.2) 
Epoch 82/125
train loss: 0.0778 - metric: 0.042 - time: 393.0 (4.6) 
valid loss: 0.1996 - metric: 0.056 - time: 393.0 (0.2) 
Epoch 83/125
train loss: 0.0759 - metric: 0.041 - time: 397.8 (4.8) 
valid loss: 0.1913 - metric: 0.053 - time: 397.8 (0.2) 
Epoch 84/125
train loss: 0.0769 - metric: 0.042 - time: 402.5 (4.8) 
valid loss: 0.2055 - metric: 0.054 - time: 402.5 (0.2) 
Epoch 85/125
train loss: 0.0763 - metric: 0.041 - time: 407.3 (4.8) 
valid loss: 0.2099 - metric: 0.055 - time: 407.3 (0.2) 
Epoch 86/125
train loss: 0.0757 - metric: 0.041 - time: 412.2 (4.8) 
valid loss: 0.2033 - metric: 0.054 - time: 412.2 (0.2) 
Epoch 87/125
train loss: 0.0756 - metric: 0.041 - time: 417.0 (4.8) 
valid loss: 0.2060 - metric: 0.054 - time: 417.0 (0.2) 
Epoch 88/125
train loss: 0.0755 - metric: 0.041 - time: 423.5 (6.5) 
valid loss: 0.2089 - metric: 0.056 - time: 423.5 (0.2) 
Epoch 89/125
train loss: 0.0748 - metric: 0.041 - time: 428.5 (5.0) 
valid loss: 0.2088 - metric: 0.053 - time: 428.5 (0.2) 
Epoch 90/125
train loss: 0.0750 - metric: 0.041 - time: 433.3 (4.9) 
valid loss: 0.2041 - metric: 0.054 - time: 433.3 (0.2) 
Epoch 91/125
train loss: 0.0749 - metric: 0.041 - time: 438.0 (4.6) 
valid loss: 0.2140 - metric: 0.057 - time: 438.0 (0.2) 
Epoch 92/125
train loss: 0.0746 - metric: 0.040 - time: 442.8 (4.8) 
valid loss: 0.2113 - metric: 0.054 - time: 442.8 (0.2) 
Epoch 93/125
train loss: 0.0746 - metric: 0.040 - time: 447.4 (4.6) 
valid loss: 0.2118 - metric: 0.055 - time: 447.4 (0.2) 
Epoch 94/125
train loss: 0.0735 - metric: 0.040 - time: 452.3 (4.9) 
valid loss: 0.2100 - metric: 0.053 - time: 452.3 (0.3) 
best
Epoch 95/125
train loss: 0.0741 - metric: 0.040 - time: 457.3 (5.0) 
valid loss: 0.2150 - metric: 0.053 - time: 457.3 (0.2) 
Epoch 96/125
train loss: 0.0737 - metric: 0.040 - time: 462.9 (5.6) 
valid loss: 0.2137 - metric: 0.053 - time: 462.9 (0.2) 
Epoch 97/125
train loss: 0.0738 - metric: 0.040 - time: 467.5 (4.6) 
valid loss: 0.2150 - metric: 0.055 - time: 467.5 (0.2) 
Epoch 98/125
train loss: 0.0736 - metric: 0.040 - time: 472.3 (4.8) 
valid loss: 0.2041 - metric: 0.054 - time: 472.3 (0.2) 
Epoch 99/125
train loss: 0.0729 - metric: 0.040 - time: 477.0 (4.7) 
valid loss: 0.2045 - metric: 0.054 - time: 477.0 (0.2) 
Epoch 100/125
train loss: 0.0726 - metric: 0.039 - time: 482.2 (5.2) 
valid loss: 0.2124 - metric: 0.053 - time: 482.2 (0.3) 
Epoch 101/125
train loss: 0.0731 - metric: 0.040 - time: 487.1 (4.9) 
valid loss: 0.2093 - metric: 0.053 - time: 487.1 (0.3) 
best
Epoch 102/125
train loss: 0.0724 - metric: 0.039 - time: 492.2 (5.0) 
valid loss: 0.2081 - metric: 0.054 - time: 492.2 (0.2) 
Epoch 103/125
train loss: 0.0729 - metric: 0.040 - time: 498.0 (5.9) 
valid loss: 0.2144 - metric: 0.056 - time: 498.0 (0.2) 
Epoch 104/125
train loss: 0.0722 - metric: 0.039 - time: 502.7 (4.7) 
valid loss: 0.2078 - metric: 0.054 - time: 502.7 (0.2) 
Epoch 105/125
train loss: 0.0721 - metric: 0.039 - time: 507.3 (4.7) 
valid loss: 0.2034 - metric: 0.053 - time: 507.3 (0.2) 
Epoch 106/125
train loss: 0.0712 - metric: 0.039 - time: 512.1 (4.7) 
valid loss: 0.2018 - metric: 0.052 - time: 512.1 (0.2) 
best
Epoch 107/125
train loss: 0.0712 - metric: 0.039 - time: 516.8 (4.7) 
valid loss: 0.2063 - metric: 0.053 - time: 516.8 (0.2) 
Epoch 108/125
train loss: 0.0717 - metric: 0.039 - time: 521.4 (4.7) 
valid loss: 0.2107 - metric: 0.053 - time: 521.4 (0.2) 
Epoch 109/125
train loss: 0.0710 - metric: 0.039 - time: 526.2 (4.8) 
valid loss: 0.2106 - metric: 0.054 - time: 526.2 (0.2) 
Epoch 110/125
train loss: 0.0719 - metric: 0.039 - time: 531.1 (4.9) 
valid loss: 0.2024 - metric: 0.052 - time: 531.1 (0.2) 
best
Epoch 111/125
train loss: 0.0707 - metric: 0.038 - time: 538.3 (7.1) 
valid loss: 0.2076 - metric: 0.054 - time: 538.3 (0.2) 
Epoch 112/125
train loss: 0.0708 - metric: 0.038 - time: 543.4 (5.2) 
valid loss: 0.2122 - metric: 0.053 - time: 543.4 (0.2) 
Epoch 113/125
train loss: 0.0703 - metric: 0.038 - time: 548.3 (4.9) 
valid loss: 0.2035 - metric: 0.053 - time: 548.3 (0.2) 
Epoch 114/125
train loss: 0.0708 - metric: 0.038 - time: 553.7 (5.3) 
valid loss: 0.2041 - metric: 0.053 - time: 553.7 (0.2) 
Epoch 115/125
train loss: 0.0703 - metric: 0.038 - time: 558.8 (5.1) 
valid loss: 0.2035 - metric: 0.053 - time: 558.8 (0.2) 
Epoch 116/125
train loss: 0.0702 - metric: 0.038 - time: 563.7 (4.8) 
valid loss: 0.2030 - metric: 0.052 - time: 563.7 (0.3) 
Epoch 117/125
train loss: 0.0701 - metric: 0.038 - time: 569.4 (5.7) 
valid loss: 0.2032 - metric: 0.052 - time: 569.4 (0.4) 
best
Epoch 118/125
train loss: 0.0698 - metric: 0.038 - time: 574.1 (4.7) 
valid loss: 0.2051 - metric: 0.053 - time: 574.1 (0.2) 
Epoch 119/125
train loss: 0.0693 - metric: 0.038 - time: 578.7 (4.6) 
valid loss: 0.2075 - metric: 0.053 - time: 578.7 (0.2) 
Epoch 120/125
train loss: 0.0695 - metric: 0.038 - time: 583.4 (4.7) 
valid loss: 0.2064 - metric: 0.052 - time: 583.4 (0.3) 
Epoch 121/125
train loss: 0.0694 - metric: 0.038 - time: 588.1 (4.7) 
valid loss: 0.2077 - metric: 0.054 - time: 588.1 (0.2) 
Epoch 122/125
train loss: 0.0697 - metric: 0.038 - time: 592.8 (4.7) 
valid loss: 0.2054 - metric: 0.053 - time: 592.8 (0.2) 
Epoch 123/125
train loss: 0.0694 - metric: 0.038 - time: 597.5 (4.8) 
valid loss: 0.2079 - metric: 0.053 - time: 597.5 (0.2) 
Epoch 124/125
train loss: 0.0692 - metric: 0.038 - time: 602.2 (4.7) 
valid loss: 0.2045 - metric: 0.053 - time: 602.2 (0.2) 
Epoch 125/125
train loss: 0.0693 - metric: 0.038 - time: 606.9 (4.7) 
valid loss: 0.2054 - metric: 0.053 - time: 606.9 (0.2) 

best epoch: 117 - metric: 0.05197009271153478 - loss: 0.2031930260283275


"""
