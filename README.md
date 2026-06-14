一，软件概述
			1. 软件用途
			FishDynamics_V1.0 是一款基于视频分析的鱼类运动力学与行为量化分析软件。软件以水下或实验视频为输入，利用深度学习图像分割模型实现鱼体目标识别，并结合多目标追踪、几何拟合与物理建模方法，对鱼类运动行为进行定量分析。
			2. 运行环境
			操作系统：Windows 10 / Windows 11
			硬件环境：CPU 环境可运行
			软件形态：Windows 可执行程序（EXE）
			3.主要功能
				1. 视频中鱼体目标的自动检测与追踪
				2. 椭圆拟合轮廓与编号可视化
				3. 距离、动能、阻力功率等指标计算
				4. 输出处理后视频、CSV 数据表与指标图
			4.输入与输出
			输入：视频文件（mp4/avi）
			输出：处理后视频、轨迹数据、统计指标图
二、软件详述
1. 研究背景与意义
			鱼类运动行为分析在水产养殖、生物行为学及水动力学研究中具有重要意义。传统人工标注或简单图像处理方法存在效率低、主观性强的问题。随着深度学习与计算机视觉技术的发展，基于视频的自动化行为分析成为研究热点。
			本软件结合深度学习图像分割模型与运动学建模方法，实现鱼类个体级别的自动识别、稳定追踪及运动指标计算，为鱼类行为定量研究提供可靠工具。
			2. 系统主要功能
				（1）基于 YOLO11-seg 的鱼体图像分割与检测
				（2）多目标追踪与个体 ID 维持
				（3）鱼体轮廓椭圆拟合与可视化
				（4）运动学与动力学指标计算
				（5）结果可视化与数据导出
			3. 程序功能实现详述
			
<img width="900" height="838" alt="image" src="https://github.com/user-attachments/assets/c23d551a-6ea6-49ef-949c-0a0f0da45476" />图1：YOLO11n-seg模型结构示意图
			（1）鱼体图像分割与检测原理
			软件采用 YOLO11-seg 图像分割模型对视频帧进行处理。该模型通过卷积神经网络学习鱼体的空间特征，实现像素级分割输出。分割结果以掩膜形式表示鱼体区域，为后续几何分析提供基础。（2）目标追踪方法
			基于检测结果，软件引入多目标追踪策略，对连续帧中的鱼体目标进行关联，为每条鱼分配唯一 ID，从而获得完整的时序运动轨迹。
			（3）椭圆拟合数学模型
			对每一帧中分割得到的鱼体轮廓点集，采用最小二乘法进行椭圆拟合。椭圆的一般方程可表示为：
			Ax² + Bxy + Cy² + Dx + Ey + F = 0
			通过约束条件求解椭圆参数，用于描述鱼体的姿态、中心位置及主轴方向。
			（4）运动学与动力学指标计算
			根据连续帧中鱼体中心位置变化，计算位移距离与速度；结合鱼体质量参数，进一步计算动能指标：
			E = 1/2 · m · v²
			同时基于流体阻力模型估算鱼体在运动过程中所受阻力功率，实现对鱼类运动状态的物理量化分析。
			（5）结果可视化与输出
			软件将分析结果以多种形式输出，包括：处理后视频（叠加椭圆与 ID 标注）、CSV 数据表、以及运动指标变化曲线图。
<img width="822" height="617" alt="image" src="https://github.com/user-attachments/assets/ee6a4d46-c915-46be-922d-051ffc72450e" />
图3：处理后视频截图
<img width="900" height="321" alt="image" src="https://github.com/user-attachments/assets/7f9b9696-b21f-4ba8-ad2d-d5bc9f6fa3ed" /><img width="900" height="563" alt="image" src="https://github.com/user-attachments/assets/11cad18b-cf47-4685-9afa-06de7f5cf2a7" />
图4：指标曲线图
软件演示效果
软件启动后，用户通过图形界面选择待分析视频并启动处理流程。系统自动完成检测、追踪与指标计算，处理结果实时保存至输出目录。
 是打开软件：双击 FishDynamics_V1.0.exe
<img width="879" height="202" alt="image" src="https://github.com/user-attachments/assets/e32d951d-cdd8-460f-807a-db5570da0227" />使用方法：第一步：手动输入：输入视频、模型权重、以及输出目录地址。
<img width="897" height="577" alt="image" src="https://github.com/user-attachments/assets/35cb9aac-92bf-4fee-8ea9-685c2d008f68" />第二步：手动输入标定参数
<img width="894" height="621" alt="image" src="https://github.com/user-attachments/assets/056028ea-8d9c-4c53-b689-5fc4bbc8daf0" />第三步：点开始处理按钮<img width="900" height="616" alt="image" src="https://github.com/user-attachments/assets/facbecc8-d6ef-4e67-8a7e-48c670483ca9" />
等待处理，处理结束后点击打开输出目录即可看到处理后的指标图、数据表格、以及处理后的视频。
<img width="900" height="539" alt="image" src="https://github.com/user-attachments/assets/e0025e27-415c-407e-b499-5b39b61e2d1a" /><img width="900" height="547" alt="image" src="https://github.com/user-attachments/assets/207173a2-285f-4e8a-8875-76ef50353710" />
同时点击，输出视频即可在界面直接看处理完成的视频；点击指标总预览即可查看指标曲线；点击总量曲线按钮即可查看总量曲线图。
<img width="900" height="539" alt="image" src="https://github.com/user-attachments/assets/0d0d95be-141a-41af-852e-eeb9f6685f0c" /><img width="900" height="656" alt="image" src="https://github.com/user-attachments/assets/1e26db7f-519c-4b7b-b06a-08aabd4ebd36" /








