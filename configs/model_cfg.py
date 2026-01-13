from easydict import EasyDict

from configs.data_cfg import cfg as data_cfg

cfg = EasyDict()

cfg.num_heads = 2
cfg.embed_dim = 8 * cfg.num_heads  # 24
cfg.num_blocks = 3
cfg.act = 'SiLU'  # SiLU

cfg.num_embed_features = (data_cfg.data_transformer.num_bins[1:] +
                          data_cfg.data_transformer.num_cats)
cfg.attn_dropout = 0.05
cfg.mlp_dropout = 0.1
cfg.dropout = 0.1
cfg.compression_factor = 0.05
cfg.compression = 'Head'  # Head KV Layer
cfg.mlp_dim_factor = 1  # 3 / 2
cfg.attn = 'Attention'  # Linear
cfg.mlp = 'GLUMLP'
cfg.norm = 'LayerNorm'
# cfg.log_softmax = False  # False True

# print(cfg.num_embed_features)

"""
cpu
Epoch 1/125
train loss: 0.6283 - metric: 0.349 - time: 4.9 (4.9) 
valid loss: 0.1868 - metric: 0.175 - time: 4.9 (0.2) 
best
Epoch 2/125
train loss: 0.1675 - metric: 0.165 - time: 9.8 (4.9) 
valid loss: 0.0713 - metric: 0.106 - time: 9.8 (0.2) 
best
Epoch 3/125
train loss: 0.0913 - metric: 0.121 - time: 14.7 (4.9) 
valid loss: 0.0541 - metric: 0.091 - time: 14.7 (0.2) 
best
Epoch 4/125
train loss: 0.0669 - metric: 0.102 - time: 19.6 (4.9) 
valid loss: 0.0498 - metric: 0.087 - time: 19.6 (0.2) 
best
Epoch 5/125
train loss: 0.0574 - metric: 0.095 - time: 24.6 (5.0) 
valid loss: 0.0493 - metric: 0.093 - time: 24.6 (0.3) 
Epoch 6/125
train loss: 0.0502 - metric: 0.088 - time: 29.4 (4.8) 
valid loss: 0.0530 - metric: 0.089 - time: 29.4 (0.2) 
Epoch 7/125
train loss: 0.0468 - metric: 0.085 - time: 34.2 (4.8) 
valid loss: 0.0356 - metric: 0.072 - time: 34.2 (0.2) 
best
Epoch 8/125
train loss: 0.0422 - metric: 0.081 - time: 39.2 (5.0) 
valid loss: 0.0440 - metric: 0.080 - time: 39.2 (0.2) 
Epoch 9/125
train loss: 0.0397 - metric: 0.079 - time: 44.1 (4.9) 
valid loss: 0.0348 - metric: 0.071 - time: 44.1 (0.2) 
best
Epoch 10/125
train loss: 0.0376 - metric: 0.076 - time: 49.0 (4.9) 
valid loss: 0.0330 - metric: 0.071 - time: 49.0 (0.2) 
Epoch 11/125
train loss: 0.0365 - metric: 0.075 - time: 53.9 (4.9) 
valid loss: 0.0386 - metric: 0.074 - time: 53.9 (0.2) 
Epoch 12/125
train loss: 0.0366 - metric: 0.075 - time: 58.8 (4.9) 
valid loss: 0.0331 - metric: 0.068 - time: 58.8 (0.2) 
best
Epoch 13/125
train loss: 0.0344 - metric: 0.073 - time: 63.8 (5.1) 
valid loss: 0.0356 - metric: 0.071 - time: 63.8 (0.3) 
Epoch 14/125
train loss: 0.0308 - metric: 0.070 - time: 68.7 (4.9) 
valid loss: 0.0297 - metric: 0.064 - time: 68.7 (0.2) 
best
Epoch 15/125
train loss: 0.0294 - metric: 0.069 - time: 73.6 (4.8) 
valid loss: 0.0313 - metric: 0.066 - time: 73.6 (0.2) 
Epoch 16/125
train loss: 0.0299 - metric: 0.069 - time: 78.3 (4.8) 
valid loss: 0.0292 - metric: 0.064 - time: 78.3 (0.2) 
best
Epoch 17/125
train loss: 0.0322 - metric: 0.072 - time: 83.2 (4.9) 
valid loss: 0.0309 - metric: 0.066 - time: 83.2 (0.3) 
Epoch 18/125
train loss: 0.0270 - metric: 0.066 - time: 88.5 (5.2) 
valid loss: 0.0292 - metric: 0.063 - time: 88.5 (0.2) 
best
Epoch 19/125
train loss: 0.0274 - metric: 0.066 - time: 93.4 (4.9) 
valid loss: 0.0321 - metric: 0.067 - time: 93.4 (0.3) 
Epoch 20/125
train loss: 0.0254 - metric: 0.064 - time: 98.2 (4.8) 
valid loss: 0.0276 - metric: 0.061 - time: 98.2 (0.2) 
best
Epoch 21/125
train loss: 0.0245 - metric: 0.064 - time: 103.1 (4.9) 
valid loss: 0.0283 - metric: 0.061 - time: 103.1 (0.2) 
Epoch 22/125
train loss: 0.0241 - metric: 0.063 - time: 107.9 (4.9) 
valid loss: 0.0318 - metric: 0.069 - time: 107.9 (0.3) 
Epoch 23/125



cpu
Epoch 1/125
train loss: 0.7265 - metric: 0.391 - time: 5.0 (5.0) 
valid loss: 0.2363 - metric: 0.217 - time: 5.0 (0.2) 
best
Epoch 2/125
train loss: 0.1605 - metric: 0.162 - time: 10.0 (5.0) 
valid loss: 0.0772 - metric: 0.105 - time: 10.0 (0.2) 
best
Epoch 3/125
train loss: 0.0903 - metric: 0.120 - time: 14.8 (4.8) 
valid loss: 0.0611 - metric: 0.103 - time: 14.8 (0.2) 
best
Epoch 4/125
train loss: 0.0685 - metric: 0.104 - time: 19.6 (4.9) 
valid loss: 0.0538 - metric: 0.089 - time: 19.6 (0.2) 
best
Epoch 5/125
train loss: 0.0580 - metric: 0.095 - time: 24.6 (4.9) 
valid loss: 0.0457 - metric: 0.085 - time: 24.6 (0.2) 
best
Epoch 6/125
train loss: 0.0527 - metric: 0.090 - time: 29.4 (4.9) 
valid loss: 0.0407 - metric: 0.079 - time: 29.4 (0.2) 
best
Epoch 7/125
train loss: 0.0481 - metric: 0.087 - time: 34.3 (4.9) 
valid loss: 0.0410 - metric: 0.079 - time: 34.3 (0.2) 
Epoch 8/125
train loss: 0.0429 - metric: 0.082 - time: 39.3 (5.0) 
valid loss: 0.0400 - metric: 0.080 - time: 39.3 (0.2) 
Epoch 9/125
train loss: 0.0400 - metric: 0.079 - time: 44.1 (4.9) 
valid loss: 0.0413 - metric: 0.079 - time: 44.1 (0.2) 
best
Epoch 10/125
train loss: 0.0376 - metric: 0.076 - time: 49.0 (4.9) 
valid loss: 0.0343 - metric: 0.072 - time: 49.0 (0.2) 
best
Epoch 11/125
train loss: 0.0365 - metric: 0.076 - time: 53.9 (4.9) 
valid loss: 0.0343 - metric: 0.070 - time: 53.9 (0.2) 
best
Epoch 12/125
train loss: 0.0339 - metric: 0.073 - time: 58.8 (4.9) 
valid loss: 0.0317 - metric: 0.069 - time: 58.8 (0.2) 
best
Epoch 13/125
train loss: 0.0325 - metric: 0.071 - time: 63.7 (4.9) 
valid loss: 0.0364 - metric: 0.073 - time: 63.7 (0.2) 
Epoch 14/125
train loss: 0.0312 - metric: 0.071 - time: 68.7 (5.0) 
valid loss: 0.0379 - metric: 0.075 - time: 68.7 (0.2) 
Epoch 15/125
train loss: 0.0300 - metric: 0.070 - time: 73.5 (4.8) 
valid loss: 0.0362 - metric: 0.073 - time: 73.5 (0.2) 
Epoch 16/125
train loss: 0.0288 - metric: 0.068 - time: 78.4 (4.9) 
valid loss: 0.0321 - metric: 0.070 - time: 78.4 (0.2) 
Epoch 17/125
train loss: 0.0279 - metric: 0.067 - time: 83.3 (4.9) 
valid loss: 0.0296 - metric: 0.067 - time: 83.3 (0.2) 
best
Epoch 18/125
train loss: 0.0277 - metric: 0.067 - time: 88.3 (5.0) 
valid loss: 0.0312 - metric: 0.068 - time: 88.3 (0.2) 
Epoch 19/125
train loss: 0.0276 - metric: 0.067 - time: 93.2 (4.9) 
valid loss: 0.0343 - metric: 0.071 - time: 93.2 (0.2) 
Epoch 20/125
train loss: 0.0274 - metric: 0.067 - time: 98.1 (4.9) 
valid loss: 0.0358 - metric: 0.072 - time: 98.1 (0.2) 
Epoch 21/125
train loss: 0.0255 - metric: 0.064 - time: 103.0 (4.9) 
valid loss: 0.0281 - metric: 0.063 - time: 103.0 (0.3) 
best
Epoch 22/125
train loss: 0.0242 - metric: 0.063 - time: 108.4 (5.4) 
valid loss: 0.0301 - metric: 0.066 - time: 108.4 (0.2) 
Epoch 23/125
train loss: 0.0240 - metric: 0.063 - time: 113.3 (5.0) 
valid loss: 0.0285 - metric: 0.064 - time: 113.3 (0.2) 
Epoch 24/125
train loss: 0.0234 - metric: 0.062 - time: 118.2 (4.9) 
valid loss: 0.0289 - metric: 0.062 - time: 118.2 (0.2) 
best
Epoch 25/125
train loss: 0.0221 - metric: 0.061 - time: 123.1 (4.9) 
valid loss: 0.0297 - metric: 0.063 - time: 123.1 (0.2) 
Epoch 26/125
train loss: 0.0225 - metric: 0.061 - time: 128.1 (4.9) 
valid loss: 0.0287 - metric: 0.062 - time: 128.1 (0.2) 
best
Epoch 27/125
train loss: 0.0210 - metric: 0.059 - time: 133.1 (5.0) 
valid loss: 0.0349 - metric: 0.072 - time: 133.1 (0.2) 
Epoch 28/125
train loss: 0.0231 - metric: 0.062 - time: 138.0 (4.9) 
valid loss: 0.0297 - metric: 0.063 - time: 138.0 (0.2) 
Epoch 29/125
train loss: 0.0230 - metric: 0.062 - time: 142.9 (4.9) 
valid loss: 0.0303 - metric: 0.065 - time: 142.9 (0.2) 
Epoch 30/125
train loss: 0.0210 - metric: 0.059 - time: 147.7 (4.9) 
valid loss: 0.0282 - metric: 0.063 - time: 147.7 (0.2) 
Epoch 31/125
train loss: 0.0223 - metric: 0.061 - time: 152.7 (4.9) 
valid loss: 0.0299 - metric: 0.063 - time: 152.7 (0.2) 
Epoch 32/125
train loss: 0.0203 - metric: 0.058 - time: 157.5 (4.9) 
valid loss: 0.0288 - metric: 0.064 - time: 157.5 (0.2) 
Epoch 33/125
train loss: 0.0200 - metric: 0.058 - time: 162.4 (4.9) 
valid loss: 0.0308 - metric: 0.064 - time: 162.4 (0.2) 
Epoch 34/125
train loss: 0.0198 - metric: 0.057 - time: 167.3 (4.9) 
valid loss: 0.0302 - metric: 0.062 - time: 167.3 (0.2) 
best
Epoch 35/125
train loss: 0.0199 - metric: 0.058 - time: 172.2 (4.9) 
valid loss: 0.0340 - metric: 0.069 - time: 172.2 (0.2) 
Epoch 36/125
train loss: 0.0216 - metric: 0.060 - time: 177.1 (4.9) 
valid loss: 0.0315 - metric: 0.064 - time: 177.1 (0.2) 
Epoch 37/125
train loss: 0.0196 - metric: 0.057 - time: 182.2 (5.0) 
valid loss: 0.0317 - metric: 0.065 - time: 182.2 (0.2) 
Epoch 38/125
train loss: 0.0185 - metric: 0.056 - time: 187.1 (4.9) 
valid loss: 0.0290 - metric: 0.063 - time: 187.1 (0.2) 
Epoch 39/125
train loss: 0.0183 - metric: 0.055 - time: 192.0 (4.9) 
valid loss: 0.0262 - metric: 0.057 - time: 192.0 (0.2) 
best
Epoch 40/125
train loss: 0.0181 - metric: 0.055 - time: 197.0 (5.0) 
valid loss: 0.0275 - metric: 0.060 - time: 197.0 (0.2) 
Epoch 41/125
train loss: 0.0171 - metric: 0.054 - time: 201.9 (4.9) 
valid loss: 0.0265 - metric: 0.059 - time: 201.9 (0.2) 
Epoch 42/125
train loss: 0.0163 - metric: 0.053 - time: 206.8 (4.9) 
valid loss: 0.0278 - metric: 0.061 - time: 206.8 (0.2) 
Epoch 43/125
train loss: 0.0174 - metric: 0.054 - time: 211.8 (5.0) 
valid loss: 0.0280 - metric: 0.061 - time: 211.8 (0.2) 
Epoch 44/125
train loss: 0.0173 - metric: 0.054 - time: 216.7 (4.9) 
valid loss: 0.0263 - metric: 0.058 - time: 216.7 (0.2) 
Epoch 45/125
train loss: 0.0169 - metric: 0.054 - time: 221.6 (4.9) 
valid loss: 0.0288 - metric: 0.061 - time: 221.6 (0.2) 
Epoch 46/125
train loss: 0.0165 - metric: 0.053 - time: 226.6 (5.1) 
valid loss: 0.0279 - metric: 0.061 - time: 226.6 (0.2) 
Epoch 47/125
train loss: 0.0166 - metric: 0.053 - time: 231.5 (4.9) 
valid loss: 0.0268 - metric: 0.058 - time: 231.5 (0.2) 
Epoch 48/125
train loss: 0.0174 - metric: 0.055 - time: 236.4 (4.9) 
valid loss: 0.0269 - metric: 0.059 - time: 236.4 (0.2) 
Epoch 49/125
train loss: 0.0160 - metric: 0.052 - time: 241.3 (4.9) 
valid loss: 0.0284 - metric: 0.062 - time: 241.3 (0.2) 
Epoch 50/125
train loss: 0.0156 - metric: 0.052 - time: 246.2 (4.9) 
valid loss: 0.0286 - metric: 0.060 - time: 246.2 (0.2) 
Epoch 51/125
train loss: 0.0156 - metric: 0.051 - time: 251.2 (5.0) 
valid loss: 0.0314 - metric: 0.065 - time: 251.2 (0.2) 
Epoch 52/125
train loss: 0.0171 - metric: 0.053 - time: 256.1 (4.9) 
valid loss: 0.0265 - metric: 0.058 - time: 256.1 (0.2) 
Epoch 53/125
train loss: 0.0167 - metric: 0.053 - time: 260.9 (4.9) 
valid loss: 0.0253 - metric: 0.056 - time: 260.9 (0.2) 
best
Epoch 54/125
train loss: 0.0155 - metric: 0.052 - time: 265.8 (4.9) 
valid loss: 0.0293 - metric: 0.061 - time: 265.8 (0.2) 
Epoch 55/125
train loss: 0.0154 - metric: 0.051 - time: 270.7 (4.9) 
valid loss: 0.0245 - metric: 0.055 - time: 270.7 (0.2) 
best
Epoch 56/125
train loss: 0.0148 - metric: 0.050 - time: 275.6 (4.9) 
valid loss: 0.0261 - metric: 0.056 - time: 275.6 (0.2) 
Epoch 57/125
train loss: 0.0138 - metric: 0.049 - time: 280.5 (4.9) 
valid loss: 0.0256 - metric: 0.057 - time: 280.5 (0.3) 
Epoch 58/125
train loss: 0.0140 - metric: 0.049 - time: 285.4 (4.9) 
valid loss: 0.0266 - metric: 0.058 - time: 285.4 (0.2) 
Epoch 59/125
train loss: 0.0142 - metric: 0.049 - time: 290.3 (4.9) 
valid loss: 0.0271 - metric: 0.057 - time: 290.3 (0.2) 
Epoch 60/125
train loss: 0.0144 - metric: 0.050 - time: 295.2 (4.9) 
valid loss: 0.0272 - metric: 0.059 - time: 295.2 (0.2) 
Epoch 61/125
train loss: 0.0137 - metric: 0.048 - time: 300.1 (4.9) 
valid loss: 0.0249 - metric: 0.055 - time: 300.1 (0.2) 
best
Epoch 62/125
train loss: 0.0140 - metric: 0.049 - time: 305.0 (4.9) 
valid loss: 0.0298 - metric: 0.062 - time: 305.0 (0.2) 
Epoch 63/125
train loss: 0.0134 - metric: 0.048 - time: 309.9 (4.9) 
valid loss: 0.0257 - metric: 0.056 - time: 309.9 (0.2) 
Epoch 64/125
train loss: 0.0135 - metric: 0.048 - time: 314.8 (4.9) 
valid loss: 0.0247 - metric: 0.054 - time: 314.8 (0.2) 
best
Epoch 65/125
train loss: 0.0133 - metric: 0.048 - time: 319.7 (4.9) 
valid loss: 0.0246 - metric: 0.054 - time: 319.7 (0.2) 
best
Epoch 66/125
train loss: 0.0129 - metric: 0.047 - time: 324.7 (5.0) 
valid loss: 0.0264 - metric: 0.057 - time: 324.7 (0.2) 
Epoch 67/125
train loss: 0.0127 - metric: 0.047 - time: 329.6 (4.9) 
valid loss: 0.0269 - metric: 0.057 - time: 329.6 (0.2) 
Epoch 68/125
train loss: 0.0130 - metric: 0.047 - time: 334.5 (4.9) 
valid loss: 0.0278 - metric: 0.058 - time: 334.5 (0.2) 
Epoch 69/125
train loss: 0.0125 - metric: 0.046 - time: 339.3 (4.9) 
valid loss: 0.0255 - metric: 0.056 - time: 339.3 (0.2) 
Epoch 70/125
train loss: 0.0125 - metric: 0.046 - time: 344.3 (4.9) 
valid loss: 0.0252 - metric: 0.056 - time: 344.3 (0.2) 
Epoch 71/125
train loss: 0.0127 - metric: 0.047 - time: 349.1 (4.9) 
valid loss: 0.0248 - metric: 0.054 - time: 349.1 (0.2) 
Epoch 72/125
train loss: 0.0123 - metric: 0.046 - time: 354.0 (4.9) 
valid loss: 0.0252 - metric: 0.055 - time: 354.0 (0.2) 
Epoch 73/125
train loss: 0.0124 - metric: 0.046 - time: 358.9 (4.9) 
valid loss: 0.0267 - metric: 0.057 - time: 358.9 (0.2) 
Epoch 74/125
train loss: 0.0121 - metric: 0.046 - time: 363.8 (4.9) 
valid loss: 0.0243 - metric: 0.054 - time: 363.8 (0.2) 
Epoch 75/125
train loss: 0.0124 - metric: 0.046 - time: 368.8 (4.9) 
valid loss: 0.0300 - metric: 0.061 - time: 368.8 (0.2) 
Epoch 76/125
train loss: 0.0121 - metric: 0.045 - time: 373.7 (5.0) 
valid loss: 0.0266 - metric: 0.056 - time: 373.7 (0.2) 
Epoch 77/125
train loss: 0.0117 - metric: 0.045 - time: 378.6 (4.9) 
valid loss: 0.0258 - metric: 0.056 - time: 378.6 (0.2) 
Epoch 78/125
train loss: 0.0113 - metric: 0.044 - time: 383.6 (5.0) 
valid loss: 0.0243 - metric: 0.054 - time: 383.6 (0.3) 
best
Epoch 79/125
train loss: 0.0115 - metric: 0.045 - time: 388.6 (5.0) 
valid loss: 0.0252 - metric: 0.054 - time: 388.6 (0.3) 
Epoch 80/125
train loss: 0.0112 - metric: 0.044 - time: 393.8 (5.2) 
valid loss: 0.0253 - metric: 0.055 - time: 393.8 (0.2) 
Epoch 81/125
train loss: 0.0109 - metric: 0.044 - time: 399.0 (5.2) 
valid loss: 0.0246 - metric: 0.054 - time: 399.0 (0.2) 
Epoch 82/125
train loss: 0.0111 - metric: 0.044 - time: 405.0 (6.1) 
valid loss: 0.0251 - metric: 0.053 - time: 405.0 (0.2) 
best
Epoch 83/125
train loss: 0.0112 - metric: 0.044 - time: 410.1 (5.0) 
valid loss: 0.0241 - metric: 0.053 - time: 410.1 (0.2) 
best
Epoch 84/125
train loss: 0.0108 - metric: 0.043 - time: 415.0 (4.9) 
valid loss: 0.0240 - metric: 0.053 - time: 415.0 (0.2) 
best
Epoch 85/125
train loss: 0.0108 - metric: 0.043 - time: 419.9 (4.9) 
valid loss: 0.0259 - metric: 0.055 - time: 419.9 (0.2) 
Epoch 86/125
train loss: 0.0106 - metric: 0.043 - time: 424.9 (5.0) 
valid loss: 0.0251 - metric: 0.055 - time: 424.9 (0.2) 
Epoch 87/125
train loss: 0.0104 - metric: 0.042 - time: 430.1 (5.1) 
valid loss: 0.0250 - metric: 0.054 - time: 430.1 (0.3) 
Epoch 88/125
train loss: 0.0106 - metric: 0.043 - time: 435.1 (5.0) 
valid loss: 0.0253 - metric: 0.055 - time: 435.1 (0.3) 
Epoch 89/125
train loss: 0.0102 - metric: 0.042 - time: 440.4 (5.3) 
valid loss: 0.0261 - metric: 0.056 - time: 440.4 (0.2) 
Epoch 90/125
train loss: 0.0102 - metric: 0.042 - time: 445.5 (5.1) 
valid loss: 0.0257 - metric: 0.055 - time: 445.5 (0.2) 
Epoch 91/125
train loss: 0.0103 - metric: 0.042 - time: 450.6 (5.1) 
valid loss: 0.0241 - metric: 0.053 - time: 450.6 (0.2) 
best
Epoch 92/125
train loss: 0.0102 - metric: 0.042 - time: 456.4 (5.8) 
valid loss: 0.0240 - metric: 0.053 - time: 456.4 (0.5) 
Epoch 93/125
train loss: 0.0099 - metric: 0.041 - time: 461.7 (5.4) 
valid loss: 0.0244 - metric: 0.053 - time: 461.7 (0.2) 
Epoch 94/125
train loss: 0.0098 - metric: 0.041 - time: 466.7 (5.0) 
valid loss: 0.0243 - metric: 0.054 - time: 466.7 (0.2) 
Epoch 95/125
train loss: 0.0097 - metric: 0.041 - time: 471.6 (4.9) 
valid loss: 0.0242 - metric: 0.053 - time: 471.6 (0.2) 
best
Epoch 96/125
train loss: 0.0097 - metric: 0.041 - time: 476.7 (5.0) 
valid loss: 0.0248 - metric: 0.054 - time: 476.7 (0.2) 
Epoch 97/125
train loss: 0.0095 - metric: 0.041 - time: 481.6 (4.9) 
valid loss: 0.0230 - metric: 0.052 - time: 481.6 (0.2) 
best
Epoch 98/125
train loss: 0.0096 - metric: 0.041 - time: 486.6 (5.0) 
valid loss: 0.0254 - metric: 0.055 - time: 486.6 (0.2) 
Epoch 99/125
train loss: 0.0094 - metric: 0.040 - time: 491.6 (5.0) 
valid loss: 0.0245 - metric: 0.054 - time: 491.6 (0.3) 
Epoch 100/125
train loss: 0.0094 - metric: 0.040 - time: 496.7 (5.1) 
valid loss: 0.0239 - metric: 0.052 - time: 496.7 (0.2) 
Epoch 101/125
train loss: 0.0094 - metric: 0.041 - time: 501.5 (4.9) 
valid loss: 0.0242 - metric: 0.053 - time: 501.5 (0.2) 
Epoch 102/125
train loss: 0.0093 - metric: 0.040 - time: 506.4 (4.9) 
valid loss: 0.0243 - metric: 0.053 - time: 506.4 (0.2) 
Epoch 103/125
train loss: 0.0092 - metric: 0.040 - time: 511.3 (4.9) 
valid loss: 0.0244 - metric: 0.053 - time: 511.3 (0.2) 
Epoch 104/125
train loss: 0.0090 - metric: 0.040 - time: 516.2 (4.9) 
valid loss: 0.0238 - metric: 0.052 - time: 516.2 (0.2) 
Epoch 105/125
train loss: 0.0092 - metric: 0.040 - time: 521.2 (5.0) 
valid loss: 0.0241 - metric: 0.053 - time: 521.2 (0.2) 
Epoch 106/125
train loss: 0.0088 - metric: 0.039 - time: 526.2 (4.9) 
valid loss: 0.0241 - metric: 0.053 - time: 526.2 (0.2) 
Epoch 107/125
train loss: 0.0090 - metric: 0.040 - time: 531.1 (4.9) 
valid loss: 0.0241 - metric: 0.052 - time: 531.1 (0.2) 
Epoch 108/125
train loss: 0.0090 - metric: 0.039 - time: 536.0 (4.9) 
valid loss: 0.0240 - metric: 0.053 - time: 536.0 (0.2) 
Epoch 109/125
train loss: 0.0090 - metric: 0.040 - time: 541.0 (5.0) 
valid loss: 0.0235 - metric: 0.052 - time: 541.0 (0.2) 
best
Epoch 110/125
train loss: 0.0090 - metric: 0.040 - time: 545.9 (4.9) 
valid loss: 0.0237 - metric: 0.052 - time: 545.9 (0.2) 
Epoch 111/125
train loss: 0.0088 - metric: 0.039 - time: 550.8 (4.9) 
valid loss: 0.0236 - metric: 0.052 - time: 550.8 (0.2) 
Epoch 112/125
train loss: 0.0087 - metric: 0.039 - time: 555.8 (5.0) 
valid loss: 0.0241 - metric: 0.053 - time: 555.8 (0.2) 
Epoch 113/125
train loss: 0.0087 - metric: 0.039 - time: 560.7 (4.9) 
valid loss: 0.0239 - metric: 0.053 - time: 560.7 (0.2) 
Epoch 114/125
train loss: 0.0087 - metric: 0.039 - time: 565.6 (4.9) 
valid loss: 0.0237 - metric: 0.052 - time: 565.6 (0.2) 
Epoch 115/125
train loss: 0.0086 - metric: 0.039 - time: 570.6 (5.0) 
valid loss: 0.0236 - metric: 0.052 - time: 570.6 (0.2) 
Epoch 116/125
train loss: 0.0087 - metric: 0.039 - time: 575.5 (4.9) 
valid loss: 0.0240 - metric: 0.052 - time: 575.5 (0.2) 
Epoch 117/125
train loss: 0.0085 - metric: 0.039 - time: 580.4 (4.9) 
valid loss: 0.0236 - metric: 0.052 - time: 580.4 (0.2) 
Epoch 118/125
train loss: 0.0087 - metric: 0.039 - time: 585.4 (5.0) 
valid loss: 0.0236 - metric: 0.052 - time: 585.4 (0.2) 
Epoch 119/125
train loss: 0.0087 - metric: 0.039 - time: 590.2 (4.9) 
valid loss: 0.0241 - metric: 0.053 - time: 590.2 (0.2) 
Epoch 120/125
train loss: 0.0085 - metric: 0.039 - time: 595.2 (4.9) 
valid loss: 0.0238 - metric: 0.052 - time: 595.2 (0.3) 
Epoch 121/125
train loss: 0.0085 - metric: 0.039 - time: 600.1 (4.9) 
valid loss: 0.0238 - metric: 0.052 - time: 600.1 (0.2) 
Epoch 122/125
train loss: 0.0086 - metric: 0.039 - time: 605.0 (4.9) 
valid loss: 0.0238 - metric: 0.052 - time: 605.0 (0.2) 
Epoch 123/125
train loss: 0.0085 - metric: 0.039 - time: 610.0 (5.0) 
valid loss: 0.0238 - metric: 0.052 - time: 610.0 (0.2) 
Epoch 124/125
train loss: 0.0086 - metric: 0.039 - time: 615.1 (5.1) 
valid loss: 0.0237 - metric: 0.052 - time: 615.1 (0.3) 
Epoch 125/125
train loss: 0.0086 - metric: 0.039 - time: 620.2 (5.0) 
valid loss: 0.0237 - metric: 0.052 - time: 620.2 (0.2) 

best epoch: 109 - metric: 0.051595395122754274 - loss: 0.02345158769887735


"""
