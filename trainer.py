import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import logging
from tqdm import tqdm
from datetime import datetime
from evaluation.SBC import sample_sbc_calstats, evaluate_sbc
from evaluation.TARP import get_ecp_area_difference
from utils import save_model, safe_update
from datasets.cos import sample_and_plot
import os
from datasets.plot_witch_hat_samples import plot_witch_hat_samples
import wandb  # Add wandb import
from nfdm_stats import NFDMStatsCollector


def trainer(data_loader, dataset, model, optimizer, scheduler, epochs, device, lr_decay, n_cal, L, seed, model_type, eval_interval, save_path, args, start_epoch=0, use_wandb=False, wandb_run=None):
    print(f"\nStarting training for {model_type} model (seed {seed})")
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize NFDM stats collector if applicable
    nfdm_stats_collector = None, nfdm_hyperparams = None
    
    if hasattr(model, 'nfdm') and hasattr(model.nfdm, 'stats_collector'):
        # Create stats collector
        nfdm_stats_collector = NFDMStatsCollector(
            save_path=save_path,
            use_wandb=use_wandb,
            wandb_run=wandb_run
        )
        
        # Set the stats collector in the model
        model.nfdm.stats_collector = nfdm_stats_collector
        
        # Extract NFDM hyperparameters
        nfdm_hyperparams = {
            'seed': seed,
            'epochs': epochs,
            'batch_size': args.batch_size,
            'n_batches': args.n_batches,
            'learning_rate': args.lr,
            'lr_decay': lr_decay,
            'dataset': args.dataset,
            'data_type': args.data_type,
            'use_encoder': args.use_encoder,
            'x_dim': model.x_dim,
            'y_dim': model.y_dim,
            'n_summaries': model.n_summaries,
            'num_hidden_layer': args.num_hidden_layer,
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'eval_interval': eval_interval,
            'n_cal': n_cal,
            'L': L,
            'ecp_n_sim': args.ecp_n_sim,
            'ecp_n_samples': args.ecp_n_samples
        }
        print(f"NFDM stats collection enabled - logging stats every {eval_interval} epochs")
        
        # Log NFDM hyperparameters to wandb if enabled
        if use_wandb and wandb_run:
            wandb_run.config.update(nfdm_hyperparams)

    evaluation_sbc = pd.DataFrame()
    loss_record = []
    training_time_record = []
    evaluation_ecp = pd.DataFrame(columns=['epoch', 'inference_time', 'ecp_score'])
    ecp_traj = pd.DataFrame()
    
    
    for epoch in tqdm(range(start_epoch, epochs), desc="Training Progress"):
        start_time = time.time()
        epoch_loss = []
        
        # Determine if we should collect stats (only during evaluation intervals)
        collect_stats_this_epoch = (epoch % eval_interval == 0) and (nfdm_stats_collector is not None)
        stats_collected = False  # Track if we've already collected stats this epoch

        for batch in data_loader:
            theta, X = batch
            if X.shape[1] == 1:
                X = X.squeeze(1)
            X = X.to(device)
            theta = theta.to(device)

            optimizer.zero_grad()
            
            # Determine if we should collect stats for this batch (only once per evaluation epoch)
            collect_stats_this_batch = collect_stats_this_epoch and not stats_collected
            stats_kwargs = None
            
            if collect_stats_this_batch:
                # Get current learning rate
                if scheduler is not None:
                    try:
                        current_lr = scheduler.get_last_lr()[0]
                    except:
                        current_lr = args.lr
                else:
                    current_lr = args.lr
                
                stats_kwargs = {
                    'epoch': epoch,
                    'seed': seed,
                    'learning_rate': current_lr,
                    'grad_norm': 0.0  # Will be updated after backward pass
                }
            
            if model_type == "NeuralDiffusionSigma" or model_type == "NeuralDiffusion":
                loss = model.loss(theta=theta, X=X, collect_stats=collect_stats_this_batch, stats_kwargs=stats_kwargs).mean() #Track NFDM stats
            else:
                loss = model.loss(theta=theta, X=X).mean() 
            epoch_loss.append(float(loss))
            
            # Mark that we've collected stats this epoch (if we did)
            if collect_stats_this_batch:
                stats_collected = True
            
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            
            # Update grad_norm in stats if we're collecting stats
            if collect_stats_this_batch and nfdm_stats_collector is not None:
                nfdm_stats_collector.current_stats['grad_norm'] = round(grad_norm.item(), 3)
            
            optimizer.step()

        if lr_decay and scheduler is not None:
            scheduler.step()

        dataset.reset_batch_sample_sizes()

        loss_record.append(np.mean(epoch_loss))
        training_time_record.append(time.time() - start_time)

        # Log basic training metrics to wandb
        if use_wandb and wandb_run:
                
            log_data = {
                'epoch': epoch,
                'train/loss': np.mean(epoch_loss),
                'train/learning_rate': current_lr,
                'train/training_time': time.time() - start_time
            }
            wandb_run.log(log_data)

        # Save loss data incrementally
        loss_df = pd.DataFrame({
            'epochs': [epoch + 1],
            'loss': [np.mean(epoch_loss)],
            'seed': [seed],
            'model_type': [model_type],
            'training_time': [time.time() - start_time]
        })
        loss_save_path = f"{save_path}/loss.csv"
        safe_update(loss_df, loss_save_path)

        if epoch % eval_interval == 0:
            inference_start_time = time.time()

            sbc_calstats = sample_sbc_calstats(dataset, n_cal, L, theta.shape[-1], model, device)
            eval_df = evaluate_sbc(sbc_calstats, seed, epoch, model_type)
            evaluation_sbc = pd.concat([evaluation_sbc, eval_df], ignore_index=True)
            
            # Log SBC metrics to wandb
            if use_wandb and wandb_run:
                # Extract SBC metrics from eval_df
                sbc_metrics = {}
                for col in eval_df.columns:
                    if col not in ['seed', 'epoch', 'model_type']:  # Skip metadata columns
                        value = eval_df[col].iloc[0]  # Get the value for this epoch
                        if pd.notna(value) and isinstance(value, (int, float)):
                            sbc_metrics[f'sbc/{col}'] = value
                
                # Log SBC metrics along with epoch
                if sbc_metrics:
                    sbc_metrics['epoch'] = epoch
                    wandb_run.log(sbc_metrics)
                    #print(f"  [Wandb] Logged {len(sbc_metrics)-1} SBC metrics for epoch {epoch}")
            
            # Save SBC data incrementally
            sbc_save_path = f"{save_path}/evaluation_sbc.csv"
            safe_update(eval_df, sbc_save_path)
            
            inference_time = time.time() - inference_start_time
            
            # Compute ecp score and ecp trajectory
            ecp_score, ecp, alpha = get_ecp_area_difference(dataset, model, device, n_sim=args.ecp_n_sim, n_samples=args.ecp_n_samples)
            
            # Log evaluation metrics to wandb
            if use_wandb and wandb_run:
                wandb_run.log({
                    'epoch': epoch,
                    'eval/ecp_score': ecp_score,
                    'eval/inference_time': inference_time
                })
                #print(f"  [Wandb] Logged evaluation metrics for epoch {epoch}")
            
            # Save ECP data incrementally
            ecp_df = pd.DataFrame({
                'epochs': [epoch],
                'inference_time': [inference_time],
                'ecp_score': [ecp_score],
                'seed': [seed],
                'model_type': [model_type]
            })
            evaluation_ecp = pd.concat([evaluation_ecp, ecp_df], ignore_index=True)
            ecp_save_path = f"{save_path}/evaluation_ecp.csv"
            safe_update(ecp_df, ecp_save_path)
            
            # Record ecp trajectory and save incrementally
            ecp_traj[f"{model_type}_epoch_{epoch}_seed_{seed}"] = ecp
            ecp_traj.index = alpha
            ecp_traj_save_path = f"{save_path}/ecp_traj.csv"
            safe_update(ecp_traj, ecp_traj_save_path, axis=1)


            ## Sample from model during training to see evolution of posterior quality ##

            # Generate cos.py plot every 100 epochs if dataset is 'cos'
            if args.dataset == "cos":
                plot_dir = os.path.join(save_path, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_name = f"estimate_posterior_{model_type}_epoch_{epoch+1}_seed_{seed}.png"
                plot_path = os.path.join(plot_dir, plot_name)
                # sample_and_plot saves multiple plots, so we set save_path to plot_dir and rely on the epoch+seed in the name
                sample_and_plot(0, model, plot_dir, device, model_type, sample_steps=100, seed=f"{seed}_epoch_{epoch+1}")
                
                # Log the plot to wandb
                if use_wandb and wandb_run:
                    try:
                        # Find the generated plot file
                        plot_files = [f for f in os.listdir(plot_dir) if f.endswith('.png') and f"epoch_{epoch+1}" in f]
                        for plot_file in plot_files:
                            plot_full_path = os.path.join(plot_dir, plot_file)
                            if os.path.exists(plot_full_path):
                                # Log the plot image to wandb
                                wandb_run.log({
                                    f"plots/{plot_file.replace('.png', '')}": wandb.Image(plot_full_path),
                                    'epoch': epoch
                                })
                                #print(f"  [Wandb] Logged plot: {plot_file}")
                    except Exception as e:
                        print(f"  [Wandb] Warning: Failed to log plot: {e}")

            # Generate witch's hat plot every 2000 epochs if dataset is 'witch_hat'
            if args.dataset == "witch_hat":
                plot_dir = os.path.join(save_path, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                
                # Generate samples for plotting
                n_samples = 150000
                X_value = 0.5
                X_dim = X.shape[-1]
                with torch.no_grad():
                    X_input = torch.full((n_samples, X_dim), X_value, dtype=torch.float32).to(device)
                    if model_type == "NormalizingFlow":
                        theta_samples = model.sample(X_input)
                    else:
                        theta_samples = model.sample(X_input, num_steps=100)
                    theta_samples = theta_samples.detach().cpu().numpy()

                # Create plot filename and save
                plot_name = f"witch_hat_{model_type}_epoch_{epoch+1}_seed_{seed}_X{X_value}.png"
                plot_path = os.path.join(plot_dir, plot_name)
                plot_witch_hat_samples(theta_samples, model_type, save_path=plot_path)
                
                # Log the plot to wandb
                if use_wandb and wandb_run:
                    try:
                        if os.path.exists(plot_path):
                            # Log the plot image to wandb
                            wandb_run.log({
                                f"plots/witch_hat_epoch_{epoch+1}": wandb.Image(plot_path),
                                'epoch': epoch
                            })
                            #print(f"  [Wandb] Logged witch's hat plot for epoch {epoch+1}")
                    except Exception as e:
                        print(f"  [Wandb] Warning: Failed to log witch's hat plot: {e}")


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
                # Repeat y_fixed for all samples
                X_input = np.tile(X_fixed, (n_samples, 1))
                X_input_torch = torch.tensor(X_input, dtype=torch.float32).to(device)
                with torch.no_grad():
                    if args.model == "NormalizingFlow":
                        theta_samples = model.sample(X_input_torch)
                    else:
                        theta_samples = model.sample(X_input_torch, num_steps=100)
                    theta_samples = theta_samples.detach().cpu()
                print(f"Theta samples shape: {theta_samples.shape}")
                # Compute the true Dirichlet posterior parameters from your data
                true_alpha_n = alpha + counts
                plot_dirichlet_margin(X_fixed[:1], theta_samples.numpy(), save_path, model_type, true_alpha_n=true_alpha_n)
            
            # Log the plot to wandb
            if use_wandb and wandb_run:
                try:
                    if os.path.exists(plot_path):
                        wandb_run.log({
                            f"plots/dirichlet_multinomial_epoch_{epoch+1}": wandb.Image(plot_path),
                            'epoch': epoch
                        })
                        #print(f"  [Wandb] Logged Dirichlet-multinomial plot for epoch {epoch+1}")
                except Exception as e:
                    print(f"  [Wandb] Warning: Failed to log Dirichlet-multinomial plot: {e}")


            # Log NFDM stats if applicable (now handled by stats collector)
            if nfdm_stats_collector is not None and nfdm_stats_collector.current_stats:
                # Log to wandb
                nfdm_stats_collector.log_to_wandb(epoch)
                
                # Save to CSV
                nfdm_stats_collector.save_to_csv()

            save_model(model, save_path, epoch, seed, model_type)
            

    print(f"\nTraining completed for {model_type} model (seed {seed})")
    print(f"Final average loss: {np.mean(loss_record[-100:]):.4f}")  # Last 100 epochs average

    # Save NFDM hyperparameters if applicable
    if nfdm_hyperparams:
        hyperparams_df = pd.DataFrame([nfdm_hyperparams])
        hyperparams_path = f"{save_path}/nfdm_hyperparams.csv"
        hyperparams_df.to_csv(hyperparams_path, index=False)
        print(f"NFDM hyperparameters saved to {hyperparams_path}")
    
    # Final save of NFDM stats if applicable
    if nfdm_stats_collector is not None:
        nfdm_stats_collector.save_to_csv()
        print(f"NFDM stats saved to {save_path}/nfdm_stats.csv")

    # Create final DataFrames for return (data already saved incrementally)
    epochs = list(range(1, len(loss_record) + 1))
    df_loss = pd.DataFrame({
        'epochs': epochs,
        'loss': loss_record,
        'seed': seed,
        'model_type': model_type,
        'training_time': training_time_record
    })

    return model, evaluation_sbc, evaluation_ecp, df_loss, ecp_traj