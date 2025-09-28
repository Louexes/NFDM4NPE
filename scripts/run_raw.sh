################################# witch hat #########################################################
#python main.py --save_path result --dataset witch_hat --device 2 --data_type=iid --epochs=2000 --model=Diffusion --save_model --eval_interval=40 --lr_decay --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &
#python main.py --save_path result --dataset witch_hat --device 2 --data_type=iid --epochs=2000 --model=NormalizingFlow --save_model --eval_interval=40  --lr_decay  --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &

######################################################## cos #########################################################
python main.py --save_path result --dataset cos --device 0 --data_type=iid --epochs=100000 --model=Diffusion --save_model --eval_interval=2000 --lr_decay --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &
python main.py --save_path result --dataset cos --device 1 --data_type=iid --epochs=100000 --model=NormalizingFlow --save_model --eval_interval=2000  --lr_decay  --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &

######################################################## socks ######################################################
#python main.py --save_path result --dataset socks --device 4 --data_type=iid --epochs=2000 --model=Diffusion --save_model --eval_interval=40   --lr_decay  --n_run=10  --ecp_n_sim=100 --ecp_n_samples=200 &
#python main.py --save_path result --dataset socks --device 4 --data_type=iid --epochs=2000 --model=NormalizingFlow --save_model --eval_interval=40   --lr_decay   --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &
#
#python main.py --save_path result --dataset species_sampling --device 6 --data_type=iid --epochs=2000 --model=Diffusion --save_model --eval_interval=40   --lr_decay  --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &
#python main.py --save_path result --dataset species_sampling --device 7 --data_type=iid --epochs=2000 --model=NormalizingFlow --save_model --eval_interval=40   --lr_decay  --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &

#python main.py --save_path result --dataset dirichlet_multinomial --device 6 --data_type=iid --epochs=2000 --model=Diffusion --save_model --eval_interval=40  --lr_decay  --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &
#python main.py --save_path result --dataset dirichlet_multinomial --device 6 --data_type=iid --epochs=2000 --model=NormalizingFlow  --save_model --eval_interval=40  --lr_decay   --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &
#
#python main.py --save_path result --dataset possion_gamma --device 7 --data_type=iid --epochs=2000 --model=Diffusion --save_model --eval_interval=40  --lr_decay  --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &
#python main.py --save_path result --dataset possion_gamma --device 7 --data_type=iid --epochs=2000 --model=NormalizingFlow --save_model --eval_interval=40  --lr_decay  --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &


python main.py --save_path result --dataset dirichlet_laplace --device 2 --data_type=iid --epochs=2000 --model=Diffusion --save_model --eval_interval=2000  --lr_decay  --n_run=1 --ecp_n_sim=100 --ecp_n_samples=200 &
python main.py --save_path result --dataset dirichlet_laplace --device 7 --data_type=iid --epochs=2000 --model=NormalizingFlow --save_model --eval_interval=40  --lr_decay  --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &



######################################################### random socks ################################################
python main.py --save_path result_random_socks --dataset socks --device 1 --data_type=iid --epochs=2000 --model=Diffusion --save_model --eval_interval=40   --lr_decay  --n_run=10  --ecp_n_sim=100 --ecp_n_samples=200 &
python main.py --save_path result_random_socks --dataset socks --device 2 --data_type=iid --epochs=2000 --model=NormalizingFlow --save_model --eval_interval=40   --lr_decay   --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200 &




##################################################### record alpha for dirichlet #######################################

python main.py --save_path result_dirichelet --dataset dirichlet_multinomial --device 0 --data_type=iid --epochs=2000 --model=Diffusion --save_model --eval_interval=2000  --lr_decay  --n_run=1 --ecp_n_sim=100 --ecp_n_samples=200 &
python main.py --save_path result_dirichelet --dataset dirichlet_multinomial --device 1 --data_type=iid --epochs=2000 --model=NormalizingFlow  --save_model --eval_interval=2000  --lr_decay   --n_run=1 --ecp_n_sim=100 --ecp_n_samples=200 &

############## rerun dirichlet for nf alpha
python main.py --save_path result_dirichelet --dataset dirichlet_multinomial --device 2 --data_type=iid --epochs=2000 --model=Diffusion --save_model --eval_interval=2000  --lr_decay  --n_run=1 --ecp_n_sim=100 --ecp_n_samples=200 --nickname=_nf_alpha &


#################################################### run 2 dim witch hat to plot #######################################

# remember to change d if you want to plot 2 dim
python main.py --save_path result_witch --dataset witch_hat --device 0 --data_type=iid --epochs=100000 --model=Diffusion --save_model --eval_interval=100000 --lr_decay --n_run=1 --ecp_n_sim=100 --ecp_n_samples=200 &
python main.py --save_path result_witch --dataset witch_hat --device 3 --data_type=iid --epochs=100000 --model=NormalizingFlow --save_model --eval_interval=100000  --lr_decay  --n_run=1 --ecp_n_sim=100 --ecp_n_samples=200 &
