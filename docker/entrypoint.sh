tmux new -d -s jupyter jupyter-lab --port 7791 --NotebookApp.token='token' --ip=0.0.0.0 --notebook-dir /home
tmux new -d -s tensorboard tensorboard --logdir /home/experiments/tb_logdir --port=7792 --host 0.0.0.0
tmux new -d -s mlflow mlflow server --backend-store-uri sqlite:////home/experiments/mlflow.db --default-artifact-root /home/experiments/artifacts --host 0.0.0.0

zsh
