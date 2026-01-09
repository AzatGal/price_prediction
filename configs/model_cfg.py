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
cfg.compression = 'KV'  # Head KV Layer
cfg.mlp_dim_factor = 1  # 5 / 3
cfg.attn = 'Attn'  # Linear
cfg.mlp = 'GLUMLP'
cfg.norm = 'LayerNorm'
cfg.log_softmax = False  # False True

# print(sum(cfg.num_embed_features))

"""
cpu
Epoch 1/125
train loss: 0.7797 - metric: 0.466 - time: 5.0 (5.0) 
valid loss: 0.6183 - metric: 0.408 - time: 5.0 (0.2) 
best
Epoch 2/125
train loss: 0.4828 - metric: 0.270 - time: 10.2 (5.2) 
valid loss: 0.3669 - metric: 0.164 - time: 10.2 (0.3) 
best
Epoch 3/125
train loss: 0.2919 - metric: 0.159 - time: 15.3 (5.1) 
valid loss: 0.3044 - metric: 0.134 - time: 15.3 (0.2) 
best
Epoch 4/125
train loss: 0.2259 - metric: 0.123 - time: 20.5 (5.2) 
valid loss: 0.2705 - metric: 0.097 - time: 20.5 (0.3) 
best
Epoch 5/125
train loss: 0.1946 - metric: 0.106 - time: 25.5 (5.1) 
valid loss: 0.2523 - metric: 0.090 - time: 25.5 (0.3) 
best
Epoch 6/125
train loss: 0.1791 - metric: 0.097 - time: 30.7 (5.1) 
valid loss: 0.2727 - metric: 0.091 - time: 30.7 (0.3) 
Epoch 7/125
train loss: 0.1748 - metric: 0.095 - time: 35.8 (5.2) 
valid loss: 0.2425 - metric: 0.079 - time: 35.8 (0.3) 
best
Epoch 8/125
train loss: 0.1634 - metric: 0.089 - time: 40.9 (5.1) 
valid loss: 0.2122 - metric: 0.075 - time: 40.9 (0.3) 
best
Epoch 9/125
train loss: 0.1537 - metric: 0.083 - time: 46.0 (5.1) 
valid loss: 0.2700 - metric: 0.084 - time: 46.0 (0.3) 
Epoch 10/125
train loss: 0.1497 - metric: 0.082 - time: 51.1 (5.1) 
valid loss: 0.2398 - metric: 0.074 - time: 51.1 (0.3) 
best
Epoch 11/125
train loss: 0.1395 - metric: 0.076 - time: 56.2 (5.1) 
valid loss: 0.2284 - metric: 0.072 - time: 56.2 (0.3) 
best
Epoch 12/125
train loss: 0.1370 - metric: 0.075 - time: 61.3 (5.0) 
valid loss: 0.2454 - metric: 0.074 - time: 61.3 (0.3) 
Epoch 13/125
train loss: 0.1335 - metric: 0.072 - time: 66.4 (5.1) 
valid loss: 0.2533 - metric: 0.069 - time: 66.4 (0.3) 
best
Epoch 14/125
train loss: 0.1329 - metric: 0.072 - time: 71.7 (5.3) 
valid loss: 0.2334 - metric: 0.071 - time: 71.7 (0.3) 
Epoch 15/125
train loss: 0.1280 - metric: 0.070 - time: 76.8 (5.1) 
valid loss: 0.2312 - metric: 0.065 - time: 76.8 (0.3) 
best

cpu
Epoch 1/125
train loss: 0.7448 - metric: 0.476 - time: 5.0 (5.0) 
valid loss: 0.5798 - metric: 0.329 - time: 5.0 (0.3) 
best
Epoch 2/125
train loss: 0.4897 - metric: 0.270 - time: 10.0 (5.0) 
valid loss: 0.3588 - metric: 0.182 - time: 10.0 (0.3) 
best
Epoch 3/125
train loss: 0.2982 - metric: 0.161 - time: 15.2 (5.1) 
valid loss: 0.2894 - metric: 0.116 - time: 15.2 (0.2) 
best
Epoch 4/125
train loss: 0.2248 - metric: 0.122 - time: 20.2 (5.1) 
valid loss: 0.2863 - metric: 0.129 - time: 20.2 (0.3) 
Epoch 5/125
train loss: 0.2161 - metric: 0.119 - time: 25.4 (5.1) 
valid loss: 0.2353 - metric: 0.087 - time: 25.4 (0.3) 
best
Epoch 6/125
train loss: 0.1754 - metric: 0.095 - time: 30.6 (5.3) 
valid loss: 0.2503 - metric: 0.092 - time: 30.6 (0.3) 
Epoch 7/125
train loss: 0.1660 - metric: 0.090 - time: 35.6 (5.0) 
valid loss: 0.2534 - metric: 0.079 - time: 35.6 (0.3) 
best
Epoch 8/125
train loss: 0.1555 - metric: 0.084 - time: 40.8 (5.1) 
valid loss: 0.2222 - metric: 0.076 - time: 40.8 (0.4) 
best
Epoch 9/125
train loss: 0.1480 - metric: 0.080 - time: 45.9 (5.2) 
valid loss: 0.2435 - metric: 0.078 - time: 45.9 (0.3) 
Epoch 10/125
train loss: 0.1399 - metric: 0.076 - time: 51.0 (5.1) 
valid loss: 0.2481 - metric: 0.073 - time: 51.0 (0.3) 
best
Epoch 11/125
train loss: 0.1333 - metric: 0.072 - time: 56.1 (5.1) 
valid loss: 0.2366 - metric: 0.071 - time: 56.1 (0.3) 
best
Epoch 12/125
train loss: 0.1312 - metric: 0.071 - time: 61.1 (5.0) 
valid loss: 0.2251 - metric: 0.064 - time: 61.1 (0.3) 
best
Epoch 13/125
train loss: 0.1263 - metric: 0.069 - time: 66.2 (5.0) 
valid loss: 0.2403 - metric: 0.068 - time: 66.2 (0.3) 
Epoch 14/125
train loss: 0.1236 - metric: 0.067 - time: 71.3 (5.2) 
valid loss: 0.2019 - metric: 0.064 - time: 71.3 (0.3) 
best
Epoch 15/125
train loss: 0.1201 - metric: 0.065 - time: 76.4 (5.1) 
valid loss: 0.1992 - metric: 0.063 - time: 76.4 (0.3) 
best
Epoch 16/125
train loss: 0.1186 - metric: 0.064 - time: 81.5 (5.1) 
valid loss: 0.2229 - metric: 0.067 - time: 81.5 (0.3) 
Epoch 17/125
train loss: 0.1164 - metric: 0.063 - time: 86.7 (5.1) 
valid loss: 0.2108 - metric: 0.063 - time: 86.7 (0.3) 
best
Epoch 18/125
train loss: 0.1143 - metric: 0.062 - time: 91.8 (5.1) 
valid loss: 0.2180 - metric: 0.063 - time: 91.8 (0.3) 
Epoch 19/125
train loss: 0.1113 - metric: 0.060 - time: 96.9 (5.1) 
valid loss: 0.2139 - metric: 0.066 - time: 96.9 (0.3) 
Epoch 20/125
train loss: 0.1095 - metric: 0.060 - time: 102.2 (5.3) 
valid loss: 0.2046 - metric: 0.063 - time: 102.2 (0.3) 
Epoch 21/125
train loss: 0.1080 - metric: 0.059 - time: 107.4 (5.2) 
valid loss: 0.2229 - metric: 0.061 - time: 107.4 (0.3) 
best
Epoch 22/125
train loss: 0.1083 - metric: 0.059 - time: 112.4 (5.1) 
valid loss: 0.2090 - metric: 0.062 - time: 112.4 (0.3) 
Epoch 23/125
train loss: 0.1062 - metric: 0.058 - time: 117.6 (5.2) 
valid loss: 0.2017 - metric: 0.060 - time: 117.6 (0.3) 
best
Epoch 24/125
train loss: 0.1041 - metric: 0.056 - time: 122.7 (5.1) 
valid loss: 0.2212 - metric: 0.059 - time: 122.7 (0.3) 
best
Epoch 25/125
train loss: 0.1019 - metric: 0.055 - time: 127.8 (5.1) 
valid loss: 0.2163 - metric: 0.059 - time: 127.8 (0.3) 
best
Epoch 26/125
train loss: 0.1008 - metric: 0.055 - time: 133.0 (5.2) 
valid loss: 0.2238 - metric: 0.063 - time: 133.0 (0.3) 
Epoch 27/125
train loss: 0.1001 - metric: 0.054 - time: 138.2 (5.2) 
valid loss: 0.2077 - metric: 0.060 - time: 138.2 (0.3) 
Epoch 28/125
train loss: 0.0988 - metric: 0.054 - time: 143.3 (5.1) 
valid loss: 0.2107 - metric: 0.057 - time: 143.3 (0.3) 
best
Epoch 29/125
train loss: 0.0975 - metric: 0.053 - time: 148.4 (5.2) 
valid loss: 0.2319 - metric: 0.063 - time: 148.4 (0.3) 
Epoch 30/125
train loss: 0.0974 - metric: 0.053 - time: 153.6 (5.1) 
valid loss: 0.1984 - metric: 0.057 - time: 153.6 (0.3) 
best
Epoch 31/125
train loss: 0.1008 - metric: 0.055 - time: 158.7 (5.1) 
valid loss: 0.2148 - metric: 0.056 - time: 158.7 (0.3) 
best
Epoch 32/125
train loss: 0.0964 - metric: 0.052 - time: 163.8 (5.2) 
valid loss: 0.2047 - metric: 0.058 - time: 163.8 (0.3) 
Epoch 33/125
train loss: 0.0952 - metric: 0.052 - time: 168.9 (5.1) 
valid loss: 0.2083 - metric: 0.058 - time: 168.9 (0.3) 
Epoch 34/125
train loss: 0.0956 - metric: 0.052 - time: 174.1 (5.2) 
valid loss: 0.2045 - metric: 0.061 - time: 174.1 (0.3) 
Epoch 35/125
train loss: 0.0927 - metric: 0.050 - time: 179.2 (5.1) 
valid loss: 0.2027 - metric: 0.057 - time: 179.2 (0.3) 
Epoch 36/125
train loss: 0.0929 - metric: 0.051 - time: 184.3 (5.1) 
valid loss: 0.2044 - metric: 0.059 - time: 184.3 (0.3) 
Epoch 37/125
train loss: 0.0912 - metric: 0.050 - time: 189.5 (5.2) 
valid loss: 0.2074 - metric: 0.061 - time: 189.5 (0.3) 
Epoch 38/125
train loss: 0.0908 - metric: 0.049 - time: 194.7 (5.1) 
valid loss: 0.2120 - metric: 0.060 - time: 194.7 (0.3) 
Epoch 39/125
train loss: 0.0906 - metric: 0.049 - time: 199.8 (5.1) 
valid loss: 0.2135 - metric: 0.057 - time: 199.8 (0.3) 
Epoch 40/125
train loss: 0.0887 - metric: 0.048 - time: 204.9 (5.2) 
valid loss: 0.1979 - metric: 0.057 - time: 204.9 (0.3) 
Epoch 41/125
train loss: 0.0881 - metric: 0.048 - time: 210.0 (5.1) 
valid loss: 0.2006 - metric: 0.059 - time: 210.0 (0.3) 
Epoch 42/125
train loss: 0.0888 - metric: 0.048 - time: 215.1 (5.1) 
valid loss: 0.2005 - metric: 0.056 - time: 215.1 (0.3) 
best
Epoch 43/125
train loss: 0.0876 - metric: 0.048 - time: 220.4 (5.2) 
valid loss: 0.1938 - metric: 0.058 - time: 220.4 (0.3) 
Epoch 44/125
train loss: 0.0893 - metric: 0.048 - time: 225.5 (5.1) 
valid loss: 0.1874 - metric: 0.056 - time: 225.5 (0.3) 
best
Epoch 45/125
train loss: 0.0854 - metric: 0.046 - time: 230.5 (5.1) 
valid loss: 0.1890 - metric: 0.054 - time: 230.5 (0.3) 
best
Epoch 46/125
train loss: 0.0860 - metric: 0.047 - time: 235.7 (5.1) 
valid loss: 0.1942 - metric: 0.056 - time: 235.7 (0.3) 
Epoch 47/125
train loss: 0.0854 - metric: 0.046 - time: 240.8 (5.1) 
valid loss: 0.1901 - metric: 0.057 - time: 240.8 (0.3) 
Epoch 48/125
train loss: 0.0854 - metric: 0.046 - time: 245.9 (5.1) 
valid loss: 0.2094 - metric: 0.058 - time: 245.9 (0.3) 
Epoch 49/125
train loss: 0.0851 - metric: 0.046 - time: 251.1 (5.2) 
valid loss: 0.1929 - metric: 0.053 - time: 251.1 (0.3) 
best
Epoch 50/125
train loss: 0.0833 - metric: 0.045 - time: 256.2 (5.1) 
valid loss: 0.1911 - metric: 0.054 - time: 256.2 (0.3) 
Epoch 51/125
train loss: 0.0825 - metric: 0.045 - time: 261.4 (5.1) 
valid loss: 0.1921 - metric: 0.055 - time: 261.4 (0.3) 
Epoch 52/125
train loss: 0.0827 - metric: 0.045 - time: 266.5 (5.1) 
valid loss: 0.1951 - metric: 0.055 - time: 266.5 (0.3) 
Epoch 53/125
train loss: 0.0826 - metric: 0.045 - time: 271.6 (5.1) 
valid loss: 0.2063 - metric: 0.058 - time: 271.6 (0.3) 
Epoch 54/125
train loss: 0.0843 - metric: 0.046 - time: 276.8 (5.2) 
valid loss: 0.1913 - metric: 0.056 - time: 276.8 (0.3) 
Epoch 55/125
train loss: 0.0814 - metric: 0.044 - time: 281.9 (5.1) 
valid loss: 0.2068 - metric: 0.055 - time: 281.9 (0.3) 
Epoch 56/125
train loss: 0.0811 - metric: 0.044 - time: 287.0 (5.1) 
valid loss: 0.1915 - metric: 0.055 - time: 287.0 (0.3) 
Epoch 57/125
train loss: 0.0811 - metric: 0.044 - time: 292.2 (5.2) 
valid loss: 0.2132 - metric: 0.059 - time: 292.2 (0.3) 
Epoch 58/125
train loss: 0.0807 - metric: 0.044 - time: 297.3 (5.1) 
valid loss: 0.2017 - metric: 0.055 - time: 297.3 (0.3) 
Epoch 59/125
train loss: 0.0807 - metric: 0.044 - time: 302.4 (5.1) 
valid loss: 0.2026 - metric: 0.055 - time: 302.4 (0.3) 
Epoch 60/125
train loss: 0.0795 - metric: 0.043 - time: 307.6 (5.2) 
valid loss: 0.1824 - metric: 0.053 - time: 307.6 (0.3) 
Epoch 61/125
train loss: 0.0785 - metric: 0.043 - time: 312.7 (5.1) 
valid loss: 0.1995 - metric: 0.054 - time: 312.7 (0.3) 
Epoch 62/125
train loss: 0.0789 - metric: 0.043 - time: 317.9 (5.2) 
valid loss: 0.1950 - metric: 0.055 - time: 317.9 (0.3) 
Epoch 63/125
train loss: 0.0784 - metric: 0.042 - time: 323.2 (5.3) 
valid loss: 0.1886 - metric: 0.054 - time: 323.2 (0.3) 
Epoch 64/125
train loss: 0.0779 - metric: 0.042 - time: 328.3 (5.1) 
valid loss: 0.1919 - metric: 0.054 - time: 328.3 (0.3) 
Epoch 65/125
train loss: 0.0779 - metric: 0.042 - time: 333.7 (5.4) 
valid loss: 0.1891 - metric: 0.055 - time: 333.7 (0.3) 
Epoch 66/125

"""
