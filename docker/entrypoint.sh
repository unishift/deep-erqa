nohup tensorboard --logdir /home/restoration-metric/tb_logdir --port=7792 --host 0.0.0.0 > /home/tb_log 2>&1 &
nohup jupyter-lab --port 7791 --NotebookApp.token='token' --ip=0.0.0.0 --notebook-dir /home > /home/jup_log 2>&1 &

zsh
