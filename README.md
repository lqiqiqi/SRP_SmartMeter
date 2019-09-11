# 实验记录

avg_loss with linear:  tensor(995.5957, device='cuda:0')

## train_0804

```
{  "exp_name": "first_train",  "model_name":"FSRCNN",  "data_dir":"../LQ_SRP_SmartMeter/data",  "num_threads":4,  "num_channels":1,  "scale_factor":5,  "num_epochs":100,  "save_epochs":10,  "batch_size":20,  "test_batch_size":1,  "save_dir":"../saving_model",  "lr":0.00001,  "gpu_mode": true,  "load_model": true}
```

训练：avg_loss = 0.7043 (MSE)

![train0804](../LQ_SRP_SmartMeter/pic/train0804.PNG)

## train0804_resnet

```
{  "exp_name": "second_train",  "model_name":"FSRCNN_resnet",  "data_dir":"../LQ_SRP_SmartMeter/data",  "num_threads":4,  "num_channels":1,  "scale_factor":5,  "num_epochs":100,  "save_epochs":10,  "batch_size":20,  "test_batch_size":1,  "save_dir":"../saving_model",  "lr":0.00001,  "gpu_mode": true,  "load_model": true}
```

训练（最后一个epoch)：avg_loss: 0.3452 (RMSE) 图中是MSE 0.11

avg_loss with original data:  tensor(121.3739, device='cuda:0') RMSE
avg_loss_log with log data:  tensor(1.3050, device='cuda:0') RMSE

![train0804_resnet](../LQ_SRP_SmartMeter/pic/train0804_resnet.PNG)

## 0805_fsrcnn_resnet_xavier

```
{  "exp_name": "third_train",  "model_name":"FSRCNN_resnet_xavier",  "data_dir":"../LQ_SRP_SmartMeter/data",  "num_threads":4,  "num_channels":1,  "scale_factor":5,  "num_epochs":100,  "save_epochs":10,  "batch_size":20,  "test_batch_size":1,  "save_dir":"../saving_model",  "lr":0.00001,  "gpu_mode": true,  "load_model": true}
```

训练（最后一个epoch)：avg_loss: 0.2098 (RMSE)

avg_loss with original data:  tensor(2346.6040, device='cuda:0') 过拟合了
avg_loss_log with log data:  tensor(0.6105, device='cuda:0')

![0805_fsrcnn_resnet_xavier](../LQ_SRP_SmartMeter/pic/0805_fsrcnn_resnet_xavier.PNG)

## 0805_2：FSRCNN_resnet_xavier_l2reg 

- 相比于上一个增加了L2reg，在SGD设置weight_decay = 1.0  

avg_loss:  0.59554976
avg_loss_log with original data:  105.55343627929688
avg_loss_log with log data:  0.5951287746429443

![0805_1_origin_test](../LQ_SRP_SmartMeter/pic/0805_1_origin_test.PNG)

![0805_1_train_test](../LQ_SRP_SmartMeter/pic/0805_1_train_test.PNG)

发生了数据泄露，因为test和train几乎重叠。检查后发现，shuffle在data_generator的init环节，这样获取train和test数据的时候都会shuffle。就可能得到同样的数据。



​			修改shuffle之后，继续使用未收敛的模型，继续训练。

​			avg_loss:  0.38546088

​			avg_loss_log with original data:  97.3842544555664
​			avg_loss_log with log data:  0.3861064016819

## 0809_1:FSRCNN_resnet_xavier_l2reg_batchsize60

- **从这里开始scale_factor改为10**

```
{  "exp_name": "6_train",  "model_name":"FSRCNN_resnet_xavier_l2reg_batchsize60",  "data_dir":"../LQ_SRP_SmartMeter/data_split",  "num_threads":4,  "num_channels":1,  "scale_factor":10,  "num_epochs":150,  "save_epochs":30,  "batch_size":60,  "test_batch_size":1,  "save_dir":"../saving_model",  "lr":0.00001,  "gpu_mode": true,  "load_model": true}
```

![0809_1_origin_test](../LQ_SRP_SmartMeter/pic/0809_1_origin_test.PNG)

![0809_1_train_test](../LQ_SRP_SmartMeter/pic/0809_1_train_test.PNG)

avg_loss:  0.97353995
avg_loss_log with original data:  108.02339935302734
avg_loss_log with log data:  0.9740539193153381
Training and test is finished.

## 0810_1：FSRCNN_resnet_xavier_l2reg_batchsize32

- 改进了加载数据的方式，使之可以利用Dataloader的subprocess的功能。
- 在前70次epoch学习率使用1\*10-4， 后80次epoch学习率1\*10-6
- 相比于上一个模型，把batchsize改为32，使用Adam优化器



只进行了30个epoch，发现loss几乎没有什么变化。

avg_loss with original data:  tensor(105.2026, device='cuda:0')
avg_loss_log with log data:  tensor(0.5026, device='cuda:0')

结论：测试结果差不多。说明减小batch_size从60到32并没有显著作用。

## 运行时间测试

可以看到np.genfromtxt()耗时很长，在神经网络那块没有什么特别好的改进方法。

```
cProfile output
--------------------------------------------------------------------------------
         62114651 function calls (62058441 primitive calls) in 28.096 seconds

   Ordered by: internal time
   List reduced from 10660 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      200    6.777    0.034   24.906    0.125 /home/jason/.local/lib/python3.6/site-packages/numpy/lib/npyio.py:1543(genfromtxt)
  6000200    3.554    0.000    6.925    0.000 /home/jason/.local/lib/python3.6/site-packages/numpy/lib/_iotools.py:236(_delimited_splitter)
  6000000    2.851    0.000    2.851    0.000 /home/jason/.local/lib/python3.6/site-packages/numpy/lib/_iotools.py:708(_loose_call)
  6000200    2.527    0.000   10.973    0.000 /home/jason/.local/lib/python3.6/site-packages/numpy/lib/_iotools.py:266(__call__)
 12006429    2.191    0.000    2.191    0.000 {method 'split' of 'str' objects}
     9113    1.567    0.000    1.567    0.000 {built-in method __new__ of type object at 0x7f9711013ce0}
  6000800    1.521    0.000    1.521    0.000 /home/jason/.local/lib/python3.6/site-packages/numpy/lib/_iotools.py:20(_decode_line)
     5654    1.337    0.000    1.337    0.000 {built-in method numpy.array}
  6006659    1.184    0.000    1.184    0.000 {method 'strip' of 'str' objects}
12064364/12063523    0.849    0.000    0.849    0.000 {built-in method builtins.len}
  6100539    0.502    0.000    0.502    0.000 {method 'append' of 'list' objects}
   114/78    0.293    0.003    0.349    0.004 {built-in method _imp.create_dynamic}
      200    0.260    0.001   25.167    0.126 /home/jason/Desktop/SRP_SmartMeter/data_loader/data_generator.py:10(load_data)
     2219    0.208    0.000    0.208    0.000 {built-in method marshal.loads}
4502/4422    0.109    0.000    0.280    0.000 {built-in method builtins.__build_class__}
```


```
--------------------------------------------------------------------------------
  autograd profiler output (CPU mode)
--------------------------------------------------------------------------------
        top 15 events sorted by cpu_time_total

------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
Name                                   CPU time        CUDA time            Calls        CPU total       CUDA total
------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
PreluBackward                        8326.570us          0.000us                1       8326.570us          0.000us
prelu_backward                       8323.527us          0.000us                1       8323.527us          0.000us
PreluBackward                        4635.385us          0.000us                1       4635.385us          0.000us
prelu_backward                       4632.611us          0.000us                1       4632.611us          0.000us
PreluBackward                        4206.233us          0.000us                1       4206.233us          0.000us
CudnnConvolutionBackward             4101.947us          0.000us                1       4101.947us          0.000us
cudnn_convolution_backward           4099.762us          0.000us                1       4099.762us          0.000us
PreluBackward                        2012.373us          0.000us                1       2012.373us          0.000us
prelu_backward                       2010.241us          0.000us                1       2010.241us          0.000us
PreluBackward                        1983.735us          0.000us                1       1983.735us          0.000us
prelu_backward                       1981.831us          0.000us                1       1981.831us          0.000us
PreluBackward                        1975.513us          0.000us                1       1975.513us          0.000us
PreluBackward                        1973.999us          0.000us                1       1973.999us          0.000us
prelu_backward                       1973.588us          0.000us                1       1973.588us          0.000us
prelu_backward                       1971.625us          0.000us                1       1971.625us          0.000us
```


```
autograd profiler output (CUDA mode)

top 15 events sorted by cpu_time_total
Because the autograd profiler uses the CUDA event API,
the CUDA time column reports approximately max(cuda_time, cpu_time).
Please ignore this output if your code does not use CUDA.

------

Name                           CPU time        CUDA time            Calls        CPU total       CUDA total

------

add                          4085.399us       4064.453us                1       4085.399us       4064.453us
UnsqueezeBackward0           4062.531us         42.969us                1       4062.531us         42.969us
PreluBackward                1994.197us        748.535us                1       1994.197us        748.535us
prelu_backward               1985.754us        745.605us                1       1985.754us        745.605us
PreluBackward                1963.171us        722.656us                1       1963.171us        722.656us
prelu_backward               1956.976us        719.727us                1       1956.976us        719.727us
PreluBackward                1950.442us        719.849us                1       1950.442us        719.849us
PreluBackward                1947.622us        728.516us                1       1947.622us        728.516us
prelu_backward               1944.592us        716.797us                1       1944.592us        716.797us
PreluBackward                1943.332us        724.609us                1       1943.332us        724.609us
prelu_backward               1940.594us        723.633us                1       1940.594us        723.633us
prelu_backward               1936.856us        721.680us                1       1936.856us        721.680us
mse_loss                     1600.102us        839.844us                1       1600.102us        839.844us
mse_loss_forward             1591.055us        835.938us                1       1591.055us        835.938us
PreluBackward                 974.085us        751.953us                1        974.085us        751.953us
```

测试了一下open()函数和list append的方法，和np.genfromtxt。发现后者特别慢，接近前者的六倍。

## 0810_3:FSRCNN_s30

- 相比于上次，把batch_size还是设为60

- 与0809_1基本是一样的，除了改进了lr的下降方式。

  ![0810_3_origin_test](../LQ_SRP_SmartMeter/pic/0810_3_origin_test.PNG)

  ![0810_3_train_test](../LQ_SRP_SmartMeter/pic/0810_3_train_test.PNG)

avg_loss:  0.46983755
avg_loss_log with original data:  87.02009582519531
avg_loss_log with log data:  0.48817479610443115

## 0810_1: FSRCNN_s32

- 相比于上次s改为32

- 加入earlystopping

  ![0811_1_origin_test](../LQ_SRP_SmartMeter/pic/0811_1_origin_test.PNG)

  ![0811_1_train_test](../LQ_SRP_SmartMeter/pic/0811_1_train_test.PNG)

  avg_loss:  0.2632
  avg_loss_log with original data:  120106
  avg_loss_log with log data:  0.3617780

  

  

## 0811_1：FSRCNN_s32

- s取32

  avg_loss:  0.252497
  avg_loss_log with original data:  2946184.0
  avg_loss_log with log data:  0.3606629967689514

  问题：test loss超级大

  ![0811_1_2_origin_test](../LQ_SRP_SmartMeter/pic/0811_1_2_origin_test.PNG)

  ![0811_1_2_train_test](../LQ_SRP_SmartMeter/pic/0811_1_2_train_test.PNG)

## 0811_2：FSRCNN_s32_explr

- s取32，lr使用指数下降。好像checkpoint中显示s为32，有点乱了。

  但是结合下一个使用explr的方法发现explr不适合。
  
  avg_loss:  0.4696042
avg_loss_log with original data:  87.22552490234375
  avg_loss_log with log data:  0.4876382648944855

  ![0811_2_origin_test](../LQ_SRP_SmartMeter/pic/0811_2_origin_test.PNG)
  
  ![0811_2_train_test](../LQ_SRP_SmartMeter/pic/0811_2_train_test.PNG)
  
## 0811_4：VDSR

- 使用lr指数下降，35epoch了loss还在0.97 0.98  0.99左右波动。果断停止，但是还是有checkpoint的。

  avg_loss with original data:  tensor(108.2328, device='cuda:0')
  avg_loss_log with log data:  tensor(1.0807, device='cuda:0')

- 改为steplr下降，到epoch30下降0.01，

avg_loss:  1.0410684
avg_loss_log with original data:  108.1648178100586
avg_loss_log with log data:  1.036773920059204



![0811_4_origin_test](../LQ_SRP_SmartMeter/pic/0811_4_origin_test.PNG)

![0811_4_train_test](../LQ_SRP_SmartMeter/pic/0811_4_train_test.PNG)





## 0812_1：FSRCNN_origin_target

- 使用s32的FSRCNN进行训练

- 使用原始数据作为target

  结果很好，说明有必要将原始数据作为target

  avg_loss:  43.25389 训练误差
  avg_loss_log with original data:  54.60212707519531 在原始数据上测试的误差

  在原始数据测试loss

  ![0812_1_origin_test](../LQ_SRP_SmartMeter/pic/0812_1_origin_test.PNG)

  

  训练loss

  ![0812_1_train](../LQ_SRP_SmartMeter/pic/0812_1_train.PNG)

## 0812_3: FSRCNN_origin_target_s12

- 相比于上个s改为12

avg_loss:  45.181908
avg_loss_log with original data:  58.62482452392578

![0812_3_origin_test](../LQ_SRP_SmartMeter/pic/0812_3_origin_test.PNG)

![0812_3_train](../LQ_SRP_SmartMeter/pic/0812_3_train.PNG)

## 0812_4: FSRCNN_origin_target_s64

- s64

  avg_loss:  42.188435
  avg_loss_log with original data:  52.23588180541992

  ![0812_4_train_test](../LQ_SRP_SmartMeter/pic/0812_4_train_test.PNG)

  



## 0812_5: FSRCNN_s12_m8

- 中间resnet添加最头和最尾的连接

avg_loss:  42.57594
avg_loss_log with original data:  61.368896484375

terminal2

![0812_5_train_test](../LQ_SRP_SmartMeter/pic/0812_5_train_test.PNG)

## 0812_6：FSRCNN_s12_m16

avg_loss:  41.85094
avg_loss_log with original data:  57.37986373901367

terminal1

![0812_6_train_test](../LQ_SRP_SmartMeter/pic/0812_6_train_test.PNG)

## 0813_1: FSRCNN_s64_m4_noshrink

terminal1

avg_loss:  96.1805
avg_loss_log with original data:  84.36612701416016

![0813_1_train_test](../LQ_SRP_SmartMeter/pic/0813_1_train_test.PNG)

## 0813_2: FSRCNN_s64_m4_noshrink

- 是上面实验去掉early stopping的再尝试

  terminal 1

  avg_loss:  100.654465
  avg_loss_log with original data:  86.87391662597656

  

## 0813_3: FSRCNN_s12_m4_fullyconnect

- 最后输出层为fullyconnect和一个卷积

avg_loss:  124.76149
avg_loss_log with original data:  108.30909729003906

![0813_3_train_test](../LQ_SRP_SmartMeter/pic/0813_3_train_test.PNG)

## 0813_4: FSRCNN_s12_m4_adagrad

- 改为adagrad下降

  avg_loss:  124.466286
  avg_loss_log with original data:  108.00846099853516

  ![0813_4_train_test](../LQ_SRP_SmartMeter/pic/0813_4_train_test.PNG)



## 0813_6：FSRCNN_s32_m4_batchsize1

avg_loss:  57.26422
avg_loss_log with original data:  54.720462799072266

用小数据测试

![0813_6_train_test](../LQ_SRP_SmartMeter/pic/0813_6_train_test.PNG)

## 0813_7: FSRCNN_s32_m4_batchsize1

用整个数据集

avg_loss:  48.98278
avg_loss_log with original data:  48.281036376953125

![0813_7_train_test](../LQ_SRP_SmartMeter/pic/0813_7_train_test.PNG)

avg_loss:  48.318565
avg_loss_log with original data:  47.672977447509766

继续这样跑200个epoch

![0813_8_train_test](../LQ_SRP_SmartMeter/pic/0813_8_train_test.PNG)



## 0815_1:FSRCNN_s32_m16_batchsize1

![0815_1_train_test](../LQ_SRP_SmartMeter/pic/0815_1_train_test.PNG)

avg_loss:  41.340076
avg_loss_log with original data:  40.87129592895508

## 0815_1:FSRCNN_s32_m8_batchsize1

![FSRCNN_s32_m8_batchsize1_train_test](../LQ_SRP_SmartMeter/pic/FSRCNN_s32_m8_batchsize1_train_test.PNG)

avg_loss:  41.70587
avg_loss_log with original data:  41.09297

## 0819_1: lr_0.1_30step



## 0819_2: lr_0.01_30step

![0819_2_train_test](../LQ_SRP_SmartMeter/pic/0819_2_train_test.PNG)



## 0820_1: upsample

![0820_1_train_test](../LQ_SRP_SmartMeter/pic/0820_1_train_test.PNG)

![0820_1_train2_test](../LQ_SRP_SmartMeter/pic/0820_1_train2_test.PNG)

继续以0.0001训练，50次后下降0.1

## 0820_2: upsample_lastconv最后一层conv

![0820_2_train_test](../LQ_SRP_SmartMeter/pic/0820_2_train_test.PNG)

## 0821_2:split10

![0821_2_train_test](../LQ_SRP_SmartMeter/pic/0821_2_train_test.PNG)

# 0901_1 不是endupsamp

![0901_1_train_test](../LQ_SRP_SmartMeter/pic/0901_1_train_test.PNG)

![0901_1_2_train_test](../LQ_SRP_SmartMeter/pic/0901_1_2_train_test.PNG)

avg_loss:  33.74135
avg_loss_log with original data:  27.338672637939453
dtw with original data:  4735.195941130432

检查后发现代码没有问题，因为nni的测试结果大部分是train loss小于test loss的。

这个出现此问题，可能是train batch size和test batch size的不同。train是8， test是1。如果train和test用一样的话，应就会小

## 0902_2: res5_endupsamp

![0902_2_train_test](../LQ_SRP_SmartMeter/pic/0902_2_train_test.PNG)

说明endupsample这个网络结构有问题

## temp

terminal4 0903_1: scale100 step50

terminal2 0903_3: batch32_step30 0.000001 step50

terminal1 0903_4: batch32_step100 0.00001 step100

terminal5 0904_1:batch8 step30 scale100 

##  0903_4: batch32_step100 0.00001 step100

![0903_4_train_test](../LQ_SRP_SmartMeter/pic/0903_4_train_test.PNG)

## 0904_1:batch8 step30 scale100 endupsamp

![0904_1_train_test](../LQ_SRP_SmartMeter/pic/0904_1_train_test.PNG)

说明结构有问题，可能是卷积upsample的问题



## temp

terminal1:  0904_3 scale10

terminal5: 0904_2 scale100

## 0904_3 scale10

![0904_3_train_test](../LQ_SRP_SmartMeter/pic/0904_3_train_test.PNG)

## 0904_2 scale100

![0904_2_train_test](../LQ_SRP_SmartMeter/pic/0904_2_train_test.PNG)

综合上面两个upsample batchsize32的结果，认为应该是网络结构有问题。先不用卷积做upsample。



## 0905_1: scale100 no upsampleconv

![0905_1_train_test](../LQ_SRP_SmartMeter/pic/0905_1_train_test.PNG)

avg_loss_log with original data:  103.1827850341

## 0905_2: scale10 no upsampleconv

![0905_2_train_test](../LQ_SRP_SmartMeter/pic/0905_2_train_test.PNG)

avg_loss_log with original data:  38.28818893432617



## temp

terminal1 10 

terminal4 100 0905_5

terminal5 10 0906_1

## 0905_4 scale10 batchsize32 noupsamp 1000epoch 

![0905_4_train_test](../LQ_SRP_SmartMeter/pic/0905_4_train_test.PNG)

Early stop at 262 epoch
avg_loss:  38.483135
avg_loss_log with original data:  38.45597457885742

## 0905_5 scale100 batchsize32 noupsamp 1000epoch 

![0905_5_train_test](../LQ_SRP_SmartMeter/pic/0905_5_train_test.PNG)



## 0906_1 scale10 batchsize32 upsamp 1000epoch 

![0906_1_train_test](../LQ_SRP_SmartMeter/pic/0906_1_train_test.PNG)

Early stop at 262 epoch
avg_loss:  34.41165
avg_loss_log with original data:  37.56041717529297

## 0906_2 scale100 batchsize32 upsamp 1000epoch 

![0906_2_train_test](../LQ_SRP_SmartMeter/pic/0906_2_train_test.PNG)

## TEMP

terminal4 0910_2 myNet batchsize8 为了和0908_2和0910_1做对比

terminal8 0908_2 liu nocrop nochop 直接把30s的数据喂进去 这个要是再不行就算了

terminal7 0910_1 用batchsize8 1晚上就达到了batchsize32 一天多的loss。因为batch size增大要求更多epoch

## 0907_1 liuNet scale100 L1

![0907_1_train_test](../LQ_SRP_SmartMeter/pic/0907_1_train_test.PNG)

使用L1做loss

avg_loss_log with original data:  111

效果很不好

## 0907_2 myNet scale100 L1

![0907_2_train_test](../LQ_SRP_SmartMeter/pic/0907_2_train_test.PNG)

没有测，但是应该效果也很差

## 0907_3 scale100 L1loss prelog

log和log做MSE loss 还是L1 loss

这样很明显会放大误差，结果很糟糕，可以不用看了

## 0907_4 liuNet scale100 L1loss prelog

## 0908_1 liu nocrop 用老的那个 切成10片

![0908_1_train_test](../LQ_SRP_SmartMeter/pic/0908_1_train_test.PNG)

不收敛

## num of params

Total number of parameters: 905984

Total number of parameters: 7260

## 0908_2 scale10 liu nocrop nochop batchsize32

![0908_2_train_test](../LQ_SRP_SmartMeter/pic/0908_2_train_test.PNG)

## 0910_1 scale10 liu nocrop nochop batchsize8

![0910_1_train_test](../LQ_SRP_SmartMeter/pic/0910_1_train_test.PNG)

batchsize 大时需要更多epoch才能达到小batchsize的效果

avg_loss:  53.076214
avg_loss_log with original data:  56.53602600097656

## 0910_2 scale10 mynet batchsize8



![0910_2_train_test](../LQ_SRP_SmartMeter/pic/0910_2_train_test.PNG)

average SNR:  11.793153934806663
avg_loss with original data:  tensor(36.0520, device='cuda:0')

## 0910_3 scale100 mynet batchsize8

![0910_3_train_test](../LQ_SRP_SmartMeter/pic/0910_3_train_test.PNG)

avg_loss:  95.13373
avg_loss_log with original data:  96.4037094116211



average SNR:  2.190822482769241
avg_loss with original data:  tensor(96.6952, device='cuda:0')



# 注意事项

1. 执行方式：在terminal中运行，注意不能加引号，argparse会自动解析为string `python main.py -c .\configs\example.json`
2. nnictl create --config config.yml
3. export PATH=$PATH:/home/jason/anaconda3/bin
4. nni search_space 注意传入不能为float，后来在base_network里面改了一下。

