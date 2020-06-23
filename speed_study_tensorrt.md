# Inference speed study

Use the lowest BS that is >= img bs

## Batch Size

BS = 1
img bs1 time: *0.037*
img bs2 time: 0.071
img bs4 time: 0.132
img bs6 time: 0.173
img bs8 time: 0.261

BS = 2
img bs1 time: 0.048
img bs2 time: *0.065*
img bs4 time: 0.122
img bs6 time: 0.179
img bs8 time: 0.245

BS = 4
img bs1 time: 0.078
img bs2 time: 0.094
img bs4 time: *0.117*
img bs6 time: 0.211
img bs8 time: 0.239

BS = 8
img bs1 time: 0.137
img bs2 time: 0.156
img bs4 time: 0.177
img bs6 time: *0.205*
img bs8 time: *0.213*
