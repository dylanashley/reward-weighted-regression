python -O ./run.py new --gamma=0.9 --epsilon=1e-9 --num-iter=50 --use-wandb=new --record-outfile=new.json
python -O ./run.py policy_iteration --gamma=0.9 --epsilon=1e-9 --num-iter=50 --use-wandb=policy_iteration --record-outfile=policy_iteration.json
