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

  

## 至今最好结果

avg_loss_log with original data:  87.02009582519531
avg_loss_log with log data:  0.48817479610443115

0810_3:FSRCNN_s30

目标21


# 注意事项

1. 执行方式：在terminal中运行，注意不能加引号，argparse会自动解析为string `python main.py -c .\configs\example.json`

