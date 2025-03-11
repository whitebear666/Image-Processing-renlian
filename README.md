# Image-Processing-renlian
从视频中提取人脸，并将清晰人脸保存
工作流：
    1.先执行main，将视频中的人脸分割出来，是用的MTCNN分析人脸，dlib GPU版太少，需要自己编译，而facenet_pytorch并不好整，重用MTCNN 进行人脸检测
    2.再执行face，使用计算图像的拉普拉斯变换方差，灰度图进行对比分析
    3.最后执行action,存储已处理图片的哈希值

    其中，只有main使用GPU进行计算，其他两个都是靠cpu进行计算
    注意，文件名称使用如"1_frame_00030"这种，用其他不行

    文件夹：XX盘
    sp --> spt --> finspt --> tp
