# 实验记录

avg_loss with linear:  tensor(995.5957, device='cuda:0')

1. train_0804

   ```
   {  "exp_name": "first_train",  "model_name":"FSRCNN",  "data_dir":"../LQ_SRP_SmartMeter/data",  "num_threads":4,  "num_channels":1,  "scale_factor":5,  "num_epochs":100,  "save_epochs":10,  "batch_size":20,  "test_batch_size":1,  "save_dir":"../saving_model",  "lr":0.00001,  "gpu_mode": true,  "load_model": true}
   ```

   训练：avg_loss = 0.7043 (MSE)

   ![train0804](../LQ_SRP_SmartMeter/pic/train0804.PNG)

   

2. train0804_resnet

   ```
   {  "exp_name": "second_train",  "model_name":"FSRCNN_resnet",  "data_dir":"../LQ_SRP_SmartMeter/data",  "num_threads":4,  "num_channels":1,  "scale_factor":5,  "num_epochs":100,  "save_epochs":10,  "batch_size":20,  "test_batch_size":1,  "save_dir":"../saving_model",  "lr":0.00001,  "gpu_mode": true,  "load_model": true}
   ```

训练（最后一个epoch)：avg_loss: 0.3452 (RMSE) 图中是MSE 0.11

avg_loss with original data:  tensor(121.3739, device='cuda:0') RMSE
avg_loss_log with log data:  tensor(1.3050, device='cuda:0') RMSE

![train0804_resnet](../LQ_SRP_SmartMeter/pic/train0804_resnet.PNG)



3. 0805_fsrcnn_resnet_xavier

   ```
   {  "exp_name": "third_train",  "model_name":"FSRCNN_resnet_xavier",  "data_dir":"../LQ_SRP_SmartMeter/data",  "num_threads":4,  "num_channels":1,  "scale_factor":5,  "num_epochs":100,  "save_epochs":10,  "batch_size":20,  "test_batch_size":1,  "save_dir":"../saving_model",  "lr":0.00001,  "gpu_mode": true,  "load_model": true}
   ```

   训练（最后一个epoch)：avg_loss: 0.2098 (RMSE)

   avg_loss with original data:  tensor(2346.6040, device='cuda:0') 过拟合了
   avg_loss_log with log data:  tensor(0.6105, device='cuda:0')

   ![0805_fsrcnn_resnet_xavier](../LQ_SRP_SmartMeter/pic/0805_fsrcnn_resnet_xavier.PNG)

   4. 0805_2：FSRCNN_resnet_xavier_l2reg 

       相比于上一个增加了L2reg，在SGD设置weight_decay = 1.0  

      avg_loss:  0.59554976
      avg_loss_log with original data:  105.55343627929688
      avg_loss_log with log data:  0.5951287746429443

      ![0805_1_origin_test](../LQ_SRP_SmartMeter/pic/0805_1_origin_test.PNG)

      ![0805_1_train_test](../LQ_SRP_SmartMeter/pic/0805_1_train_test.PNG)

      发生了数据泄露，因为test和train几乎重叠。检查后发现，shuffle在data_generator的init环节，这样获取train和test数据的时候都会shuffle。就可能得到同样的数据。

# 注意事项

1. 执行方式：在terminal中运行，注意不能加引号，argparse会自动解析为string `python main.py -c .\configs\example.json`

