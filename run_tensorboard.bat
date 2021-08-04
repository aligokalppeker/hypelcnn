set CONDA_PATH=C:\Users\Ali.Peker\Miniconda3\envs\phd
%set CONDA_PATH=C:\ProgramData\Anaconda3\envs\tensorflow
PATH=%PATH%;%CONDA_PATH%;%CONDA_PATH%\Library\mingw-w64\bin;%CONDA_PATH%\Library\usr\bin;%CONDA_PATH%\Library\bin;%CONDA_PATH%\Scripts;
start http://agp-pc:6006
@echo tensorboard --logdir=successful_logs
tensorboard --logdir=log