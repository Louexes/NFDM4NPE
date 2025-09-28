################################################# DepSet ##############################################################
python main.py --save_path result --dataset g_and_k --device 2 --data_type=iid --epochs=5000 --model=Diffusion --use_encoder --save_model --eval_interval=100 --lr_decay  --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 --nickname=5k &
python main.py --save_path result --dataset g_and_k --device 3 --data_type=iid --epochs=5000 --model=NormalizingFlow --use_encoder --save_model --eval_interval=100  --lr_decay --n_run=10   --ecp_n_sim=100 --ecp_n_samples=200 --nickname=5k &

#python main.py --save_path result --dataset normal_gamma --device 1 --data_type=iid --epochs=2000 --model=Diffusion --use_encoder  --save_model --eval_interval=40 --lr_decay --n_run=10   --ecp_n_sim=100 --ecp_n_samples=200 &
#python main.py --save_path result --dataset normal_gamma --device 1 --data_type=iid --epochs=2000 --model=NormalizingFlow --use_encoder --save_model --eval_interval=40 --lr_decay --n_run=10   --ecp_n_sim=100 --ecp_n_samples=200 &
#
#python main.py --save_path result --dataset normal_wishart --device 2 --data_type=iid --epochs=5000 --model=Diffusion --use_encoder  --save_model --eval_interval=40 --lr_decay  --n_run=10  --ecp_n_sim=100 --ecp_n_samples=200 &
#python main.py --save_path result --dataset normal_wishart --device 2 --data_type=iid --epochs=5000 --model=NormalizingFlow --use_encoder --save_model --eval_interval=40 --lr_decay  --n_run=10  --ecp_n_sim=100 --ecp_n_samples=200 &
#

################################################# Transformer ##########################################################
#
#python main.py --save_path result --dataset g_and_k --device 2 --data_type=set --epochs=2000 --model=Diffusion --use_encoder --save_model --eval_interval=100 --lr_decay  --n_run=10  --ecp_n_sim=100 --ecp_n_samples=200 --batch_size=64 --n_batch=256 --nickname=set &
#python main.py --save_path result --dataset g_and_k --device 3 --data_type=set --epochs=2000 --model=NormalizingFlow --use_encoder --save_model --eval_interval=100  --lr_decay --n_run=10   --ecp_n_sim=100 --ecp_n_samples=200  --nickname=set &
#
#python main.py --save_path result_set --dataset normal_gamma --device 0 --data_type=set --epochs=2000 --model=Diffusion --use_encoder  --save_model --eval_interval=40 --lr_decay --n_run=10   --ecp_n_sim=100 --ecp_n_samples=200  --nickname=set &
#python main.py --save_path result_set --dataset normal_gamma --device 1 --data_type=set --epochs=2000 --model=NormalizingFlow --use_encoder --save_model --eval_interval=40 --lr_decay --n_run=10   --ecp_n_sim=100 --ecp_n_samples=200  --nickname=set &
#
#python main.py --save_path result_set --dataset normal_wishart --device 2 --data_type=set --epochs=5000 --model=Diffusion --use_encoder  --save_model --eval_interval=40 --lr_decay  --n_run=10  --ecp_n_sim=0 --ecp_n_samples=0  --nickname=set &
#python main.py --save_path result_set --dataset normal_wishart --device 3 --data_type=set --epochs=5000 --model=NormalizingFlow --use_encoder --save_model --eval_interval=40 --lr_decay  --n_run=10  --ecp_n_sim=0 --ecp_n_samples=0  --nickname=set &
#



################################################### test ###############################################################

#python main.py --save_path test --dataset g_and_k --device 2 --data_type=iid --epochs=1 --model=Diffusion --use_encoder --save_model --eval_interval=5000 --lr_decay  --n_run=1 --ecp_n_sim=100 --ecp_n_samples=200 --nickname=5k&
#python main.py --save_path test --dataset g_and_k --device 3 --data_type=iid --epochs=5000 --model=NormalizingFlow --use_encoder --save_model --eval_interval=5000  --lr_decay --n_run=1   --ecp_n_sim=100 --ecp_n_samples=200 &
