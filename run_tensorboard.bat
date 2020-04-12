PATH=%PATH%;C:\ProgramData\Anaconda3\envs\tensorflow;C:\ProgramData\Anaconda3\envs\tensorflow\Library\mingw-w64\bin;C:\ProgramData\Anaconda3\envs\tensorflow\Library\usr\bin;C:\ProgramData\Anaconda3\envs\tensorflow\Library\bin;C:\ProgramData\Anaconda3\envs\tensorflow\Scripts;
start http://AGP-PC:6006
@echo tensorboard --logdir=utilities\log
tensorboard --logdir=successful_logs
@echo tensorboard --logdir=log
@echo tensorboard --logdir=successful_logs\grss2018\spectral_1_spatial_1_degrade_9\cnnv4_run_accuracy_0959_1x1
@echo tensorboard --logdir=successful_logs\gulfport\spectral_3_spatial_3_degrade_3\cnnv4_run_accuracy_0949_2x2_itr35k