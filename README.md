# 2Dto3D
mono23d.py xz师兄的fast-inpaint方法
mono2stereo_lowres.py zc师兄的降分辨率处理后再上采样的方法
mono2stereo_lower_fastinpaint.py 在xz师兄的框架上加入了降分辨率上采样的方法

由于zc师兄的方法好像没有统计opencv写视频的时间，因此比较了mono23d和mono2stereo_lower_fastinpaint

前者16.6帧，后者26.1帧。

在民大的3090（第七张卡上测试）：
mono2stereo_lower_fastinpaint的方法只能跑10帧（感觉可能跟文件编码格式有关，ffmep那个，还有cpu调用？）
mono2stereo_lower_fastinpaint_time运行时额外加入--profile-time即可计时（显示界面优化）
然后因为服务器这两天连不上，故而花了点时间配了下环境和权重，先进行了gpu方面的优化。
mono2stereo_lower_fastinpaint_time_gpu
1.颜色转换移到 GPU
2.多线程预读队列（CPU 和 GPU 同时工作）
3.预处理全部 GPU 化（MyResize、Normalize 全移到 GPU）
速度从10帧提升到了16帧。
可以运行mono2stereo_lower_fastinpaint_time和mono2stereo_lower_fastinpaint_time_gpu进行对比。（加参数--profile-time）