python main.py --save_path result --dataset lotka_volterra --use_encoder --device 0 --data_type=time --epochs=5000 --model=Diffusion --lr_decay  --eval_interval=100 --n_run=10 --save_model --ecp_n_sim=100 --ecp_n_samples=200 &
python main.py --save_path result --dataset lotka_volterra --use_encoder --device 1 --data_type=time --epochs=5000 --model=NormalizingFlow --lr_decay  --eval_interval=100 --n_run=10  --save_model  --ecp_n_sim=100 --ecp_n_samples=200 &

python main.py --save_path result --dataset markov_switch --use_encoder --device 2 --data_type=time --epochs=5000 --model=Diffusion --lr_decay  --eval_interval=100  --n_run=10  --save_model  --ecp_n_sim=100 --ecp_n_samples=200 &
python main.py --save_path result --dataset markov_switch --use_encoder --device 3 --data_type=time --epochs=5000 --model=NormalizingFlow --lr_decay  --eval_interval=100  --n_run=10  --save_model  --ecp_n_sim=100 --ecp_n_samples=200 &

python main.py --save_path result --dataset stochastic_vol --use_encoder --device 4 --data_type=time --epochs=5000 --model=Diffusion --lr_decay --eval_interval=100   --n_run=10  --save_model  --ecp_n_sim=100 --ecp_n_samples=200 &
python main.py --save_path result --dataset stochastic_vol --use_encoder --device 5 --data_type=time --epochs=5000 --model=NormalizingFlow --lr_decay --eval_interval=100  --n_run=10  --save_model  --ecp_n_sim=100 --ecp_n_samples=200 &

python main.py --save_path result --dataset fBM --use_encoder --device 6 --data_type=time --epochs=5000 --model=Diffusion --lr_decay  --eval_interval=100  --n_run=10  --save_model  --ecp_n_sim=100 --ecp_n_samples=200 &
python main.py --save_path result --dataset fBM --use_encoder --device 7 --data_type=time --epochs=5000 --model=NormalizingFlow --lr_decay  --eval_interval=100   --n_run=10  --save_model  --ecp_n_sim=100 --ecp_n_samples=200 &

python main.py --save_path result --dataset minnesota --use_encoder --device 2 --data_type=time --epochs=5000 --model=Diffusion --lr_decay  --eval_interval=100  --n_run=10  --save_model  --ecp_n_sim=100 --ecp_n_samples=200 &
python main.py --save_path result --dataset minnesota --use_encoder --device 1 --data_type=time --epochs=5000 --model=NormalizingFlow --lr_decay  --eval_interval=100   --n_run=10  --save_model  --ecp_n_sim=100 --ecp_n_samples=200 &


######################################### save more checkpoint #######################################################

python main.py --save_path result_lotka --dataset lotka_volterra --use_encoder --device 2 --data_type=time --epochs=5000 --model=Diffusion --lr_decay  --eval_interval=100 --n_run=1 --save_model --ecp_n_sim=100 --ecp_n_samples=200 &
python main.py --save_path result_lotka --dataset lotka_volterra --use_encoder --device 2 --data_type=time --epochs=5000 --model=NormalizingFlow --lr_decay  --eval_interval=100 --n_run=1  --save_model  --ecp_n_sim=100 --ecp_n_samples=200 &

python main.py --save_path result_lotka --dataset fBM --use_encoder --device 2 --data_type=time --epochs=5000 --model=Diffusion --lr_decay  --eval_interval=100 --n_run=1 --save_model --ecp_n_sim=100 --ecp_n_samples=200 &
python main.py --save_path result_lotka --dataset fBM --use_encoder --device 1 --data_type=time --epochs=5000 --model=NormalizingFlow --lr_decay  --eval_interval=100 --n_run=1  --save_model  --ecp_n_sim=100 --ecp_n_samples=200 &

