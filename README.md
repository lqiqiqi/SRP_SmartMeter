# 实验记录

1. ```
   {  "exp_name": "first_train",  "model_name":"FSRCNN",  "data_dir":"../LQ_SRP_SmartMeter/data",  "num_threads":4,  "num_channels":1,  "scale_factor":5,  "num_epochs":100,  "save_epochs":10,  "batch_size":20,  "test_batch_size":1,  "save_dir":"../saving_model",  "lr":0.00001,  "gpu_mode": true,  "load_model": true}
   ```

   avg_loss = 0.7043 (MSE)

   ![train0804](../LQ_SRP_SmartMeter/pic/train0804.PNG)

   



# 注意事项

1. 执行方式：在terminal中运行，注意不能加引号，argparse会自动解析为string `python main.py -c .\configs\example.json`

