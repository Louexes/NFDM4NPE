import torch
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from models.neural_sampler import NormalizingFlowPosteriorSampler, DiffusionPosteriorSampler, NeuralDiffusionPosteriorSamplerSigma, NeuralDiffusionPosteriorSampler
from evaluation.SBC import sample_sbc_calstats, evaluate_sbc
from evaluation.TARP import get_ecp_area_difference
from utils import *
import pandas as pd
import time
import argparse
import os
from trainer import trainer
import wandb  # Add wandb import



def main(args):
    print("\n=== Training Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model Type: {args.model}")
    print(f"Number of Runs: {args.n_run}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Batches: {args.n_batches}")
    print(f"Learning Rate: {args.lr}")
    print(f"Learning Rate Decay: {args.lr_decay}")
    print(f"Evaluation Interval: {args.eval_interval}")
    print(f"Device: {args.device}")
    print("===========================\n")


    # Dataset paramaters
    n_batches = args.n_batches
    batch_size = args.batch_size

    # Model paramaters
    hidden_dim_summary_net = 32
    n_summaries = 256  # sufficient statistics for normal-gamma model
    DEVICE = args.device
    alpha = args.alpha
    use_encoder = bool(args.use_encoder)
    n_sample = None if use_encoder else 1

    # Opitimzer paramaters
    epochs = args.epochs
    lr = args.lr
    lr_decay = args.lr_decay

    # Evaluate paramaters
    n_cal, L, model_type = args.n_cal, args.L, args.model

    if args.nickname is not None:
        model_type += args.nickname

    n_run = args.n_run
    eval_interval = args.eval_interval

    print("Loading dataset...")
    dataset_generator, sample_theta, sample_data = load_dataset(args.dataset)

    if args.use_encoder:
        dl = dataset_generator(n_batches, batch_size, return_ds=False)
    else:
        dl = dataset_generator(n_batches, batch_size, n_sample=1, return_ds=False)
    theta, X = next(iter(dl))
    X_dim = X.shape[-1]
    theta_dim = theta.shape[1]

    print(f"Dataset loaded. Input dimension: {X_dim}, Parameter dimension: {theta_dim}")

    for i in range(1,n_run+1):
        print(f"\n=== Starting Run {i}/{n_run} ===")
        seed = i + args.seed_start
        SET_SEED(seed)
        print(f"Using seed: {seed}")

        # Initialize wandb run
        if args.use_wandb:
            run_name = f"{args.model}_{args.save_path}_{args.dataset}_seed{seed}"
            wandb_run = wandb.init(
                entity="fomo-ai4math",
                project="NFDM",
                name=run_name,
                config={
                    'model_type': args.model,
                    'dataset': args.dataset,
                    'seed': seed,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'lr_decay': args.lr_decay,
                    'n_batches': args.n_batches,
                    'alpha': args.alpha,
                    'use_encoder': args.use_encoder,
                    'num_hidden_layer': args.num_hidden_layer,
                    'n_cal': args.n_cal,
                    'L': args.L,
                    'eval_interval': args.eval_interval,
                },
                tags=[args.model, args.dataset, f"seed_{seed}"]
            )
            print(f"Wandb run initialized: {wandb_run.name}")
        else:
            wandb_run = None

        if args.model == "NormalizingFlow":
            model = NormalizingFlowPosteriorSampler(X_dim=X_dim, theta_dim=theta_dim, n_summaries=n_summaries,
                                       hidden_dim_decoder=hidden_dim_summary_net, n_flows_decoder=32, alpha=alpha, device=DEVICE,
                                       use_encoder=use_encoder, data_type=args.data_type).to(DEVICE)
        elif args.model == "Diffusion":
            if args.use_emperical_sigma:
                sigma_data = theta.std().item()
            else:
                sigma_data = 0.5
            num_hidden_layer = args.num_hidden_layer
            model = DiffusionPosteriorSampler(X_dim=X_dim, theta_dim=theta_dim, n_summaries=n_summaries,num_hidden_layer=num_hidden_layer,
                                              device=DEVICE,use_encoder=use_encoder, data_type=args.data_type, sigma_data=sigma_data)

        elif args.model == "NeuralDiffusionSigma":
            if args.use_emperical_sigma:
                sigma_data = theta.std().item()
            else:
                sigma_data = 0.5
            print("Initializing NeuralDiffusionSigma model...")
            num_hidden_layer = args.num_hidden_layer
            model = NeuralDiffusionPosteriorSamplerSigma(X_dim=X_dim, theta_dim=theta_dim, n_summaries=n_summaries, num_hidden_layer=num_hidden_layer,
                                              device=DEVICE,use_encoder=use_encoder, data_type=args.data_type, sigma_data=sigma_data)

        elif args.model == "NeuralDiffusion":
            num_hidden_layer = args.num_hidden_layer
            model = NeuralDiffusionPosteriorSampler(X_dim=X_dim, theta_dim=theta_dim, n_summaries=n_summaries, num_hidden_layer=num_hidden_layer,
                                              device=DEVICE,use_encoder=use_encoder, data_type=args.data_type)
        else:
            raise NotImplementedError


        save_path = f"{args.save_path}/{args.dataset}/{args.model}"
        print(f"Results will be saved to: {save_path}")
        os.makedirs(save_path, exist_ok=True)


        dl, ds = dataset_generator(n_batches, batch_size, n_sample, return_ds=True)

        if args.load_model:
            print(f"Loading model from checkpoint...")
            # Use specified checkpoint epoch or default to final epoch
            checkpoint_epoch = args.checkpoint_epoch if args.checkpoint_epoch is not None else epochs
            model = load_torch_model(model, save_path, checkpoint_epoch, seed, model_type)
            
            if args.checkpoint_epoch is not None:
                # Continue training from specified checkpoint
                print(f"Resuming training from epoch {checkpoint_epoch}...")
                # Initialize optimizer and scheduler for continued training
                print("Initializing optimizer and scheduler...")
                
                optimizer = optim.Adam(model.parameters(), lr=lr)
                optimizer_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
                
                # Training (resume from checkpoint)
                model, evaluation_sbc, evaluation_ecp, df_loss, ecp_traj = trainer(
                    dl, ds, model, optimizer, optimizer_sched, epochs, DEVICE, True, 
                    n_cal, L, seed, model_type, eval_interval, save_path, args, 
                    start_epoch=checkpoint_epoch, use_wandb=args.use_wandb, wandb_run=wandb_run
                )
            else:
                # Evaluation only (no checkpoint_epoch specified)
                print(f"Evaluation only mode - no training will be performed")
                # Initialize empty DataFrames for evaluation only
                evaluation_sbc = pd.DataFrame()
                evaluation_ecp = pd.DataFrame()
                df_loss = pd.DataFrame()
                ecp_traj = pd.DataFrame()
        else:
            # Fresh training
            print("Initializing optimizer and scheduler...")
            print(f"Fresh training: {epochs} epochs with learning rate {lr}")
            optimizer = optim.Adam(model.parameters(), lr=lr)
            optimizer_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            # Training (start fresh)
            model, evaluation_sbc, evaluation_ecp, df_loss, ecp_traj = trainer(
                dl, ds, model, optimizer, optimizer_sched, epochs, DEVICE, lr_decay, 
                n_cal, L, seed, model_type, eval_interval, save_path, args,
                use_wandb=args.use_wandb, wandb_run=wandb_run
            )
            

        if args.save_model:
            print("Saving final model...")
            save_model(model, save_path, epochs, seed, model_type)
            

        # All metrics are already saved incrementally during training
        print("All training metrics were saved incrementally during training")

        # Print total and trainable parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        ## Final Evaluation
        print("\nPerforming final evaluation...")
        sbc_calstats = sample_sbc_calstats(ds, n_cal, L, theta_dim, model, DEVICE)
        eval_df = evaluate_sbc(sbc_calstats, seed, epochs, model_type)
        evaluation_sbc = pd.concat([evaluation_sbc, eval_df], ignore_index=True)
        
        evaluation_sbc_save_path = f"{save_path}/evaluation_sbc.csv"
        evaluation_ecp_save_path = f"{save_path}/evaluation_ecp.csv"
        df_loss_save_path = f"{save_path}/loss.csv"
        ecp_traj_save_path = f"{save_path}/ecp_traj.csv"
        safe_update(evaluation_sbc, evaluation_sbc_save_path)
        safe_update(evaluation_ecp, evaluation_ecp_save_path)
        safe_update(df_loss, df_loss_save_path)
        safe_update(ecp_traj, ecp_traj_save_path, axis=1)


        # plot_hist(sbc_calstats,save_path,seed,model_type)
        if args.dataset in ["socks", "species_sampling","dirichlet_laplace"]:
            plot_scatter(X.squeeze(1),theta,model,save_path,seed,model_type,DEVICE)

        if args.dataset == "cos":
            from datasets.cos import plot_posterior, sample_and_plot
            plot_posterior(X_observed = 0)
            sample_and_plot(0,model,save_path, DEVICE, model_type, sample_steps=100, seed=seed)
            
            # Log final cos plots to wandb
            if args.use_wandb and wandb_run:
                try:
                    plot_dir = os.path.join(save_path, "plots")
                    if os.path.exists(plot_dir):
                        plot_files = [f for f in os.listdir(plot_dir) if f.endswith('.png') and f"seed{seed}" in f]
                        for plot_file in plot_files:
                            plot_full_path = os.path.join(plot_dir, plot_file)
                            wandb_run.log({
                                f"final_plots/cos_{plot_file.replace('.png', '')}": wandb.Image(plot_full_path)
                            })
                            print(f"[Wandb] Logged final cos plot: {plot_file}")
                except Exception as e:
                    print(f"[Wandb] Warning: Failed to log final cos plot: {e}")

        if args.dataset == "dirichlet_multinomial":
            print("Generating Dirichlet-multinomial marginal plots...")
            from datasets.dirichlet_multinomial import plot_dirichlet_margin, n_multi
            # Use symmetric Dirichlet prior
            alpha = np.ones(5)
            N = 315
            p_hat = np.array([0.0667, 0.1600, 0.2367, 0.2200, 0.3167])
            counts = np.round(N * p_hat).astype(int)
            # Adjust counts to sum to N if needed
            diff = N - counts.sum()
            if diff != 0:
                # Add/subtract the difference to the largest/smallest element
                idx = np.argmax(counts) if diff > 0 else np.argmin(counts)
                counts[idx] += diff
            X_fixed = counts / N  # proportions, shape (5,)
            n_samples = 315
            # Repeat X_fixed for all samples
            X_input = np.tile(X_fixed, (n_samples, 1))
            X_input_torch = torch.tensor(X_input, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                if args.model == "NormalizingFlow":
                    theta_samples = model.sample(X_input_torch)
                else:
                    theta_samples = model.sample(X_input_torch, num_steps=100)
                theta_samples = theta_samples.detach().cpu()
            print(f"Theta samples shape: {theta_samples.shape}")
            # Compute the true Dirichlet posterior parameters from your data
            true_alpha_n = alpha + counts
            plot_dirichlet_margin(X_input[:1], theta_samples.numpy(), save_path, model_type, true_alpha_n=true_alpha_n)

    
        if args.dataset == "witch_hat":
            print("Generating Witch's Hat sample plots...")
            from datasets.plot_witch_hat_samples import plot_witch_hat_samples
            # Generate samples from the model for plotting
            n_samples = 150000
            # Set a specific y value for plotting (e.g., y = 0.5)
            X_value = 0.5
            
            with torch.no_grad():
                # Create input tensor with the specified y value
                X_input = torch.full((n_samples, X_dim), X_value, dtype=torch.float32).to(DEVICE)
                if args.model == "NormalizingFlow":
                    theta_samples = model.sample(X_input)
                else:
                    theta_samples = model.sample(X_input, num_steps=100)
                theta_samples = theta_samples.detach().cpu().numpy()
            print(f"Theta samples shape: {theta_samples.shape}")
            # Plot the samples
            plot_save_path = f"{save_path}/witch_hat_{model_type}_seed{seed}_X{X_value}.png"
            plot_witch_hat_samples(theta_samples, model_type, save_path=plot_save_path)
            
            # Log final witch's hat plot to wandb
            if args.use_wandb and wandb_run:
                try:
                    if os.path.exists(plot_save_path):
                        wandb_run.log({
                            f"final_plots/witch_hat_final": wandb.Image(plot_save_path)
                        })
                        print(f"[Wandb] Logged final witch's hat plot")
                except Exception as e:
                    print(f"[Wandb] Warning: Failed to log final witch's hat plot: {e}")

    
        # Close wandb run
        if args.use_wandb and wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple example of argparse")

    ## Training parameters
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--model', type=str, default="NormalizingFlow", help="NormalizingFlow or Diffusion")
    parser.add_argument('--n_run', type=int, default=1, help="How many runs to repeat")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', action='store_true',)
    parser.add_argument('--n_batches', type=int, default=2, help="Number of batches for an epoch")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_path', type=str, default="./test")
    parser.add_argument('--alpha', type=float, default=0.1, help="Parameter for normalizing flow to control Lipschitz constant.")
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--use_encoder', action='store_true', help="Use summary network or not")
    parser.add_argument('--use_emperical_sigma', action='store_true', help="whether to set \sigma_data as empirical std of data, otherwise 0.5 as EDM")
    parser.add_argument('--num_hidden_layer',type=int, default=4, help="Number of hidden layers for diffusion model")

    ## Dataset parameters
    parser.add_argument('--dataset', type=str, default="dirichlet_multinomial", help="Please see all datasets name in datasets/__init__.py")
    parser.add_argument('--data_type', type=str, default="iid", help="iid or time")

    ## Evaluation parameters
    parser.add_argument('--n_cal', type=int, default=1000, help="Number of calibration for SBC")
    parser.add_argument('--L', type=int, default=100, help="Number of posterior samples per x for SBC, same notation with SBC paper")
    parser.add_argument('--ecp_n_sim', type=int, default=1000, help="Number of simulations for TARP")
    parser.add_argument('--ecp_n_samples', type=int, default=2000, help="Number of posterior samples per x for TARP")
    
    # Utility parameters
    parser.add_argument('--save_model', action='store_true', help="Use encoder or not")
    parser.add_argument('--load_model', action='store_true', help="Load model from checkpoint")
    parser.add_argument('--checkpoint_epoch', type=int, default=None, help="Epoch number of checkpoint to load (defaults to final epoch)")

    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--nickname', type=str, default=None, help="Add a nickname to the save folder")
    parser.add_argument('--seed_start', type=int, default=0)

    # Add wandb arguments
    parser.add_argument('--use_wandb', default=True, action='store_true', help="Enable Weights & Biases logging")

    args = parser.parse_args()
    main(args)


