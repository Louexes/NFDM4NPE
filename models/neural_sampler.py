import torch
from .summary import DeepSetSummary, BayesFlowEncoder, SetEmbedderClean
from .normalizing_flow import ConditionalNormalizingFlow
from .diffusion import *
from .nfdm_sigma import *
from .nfdm_t import *

## NFDM (Sigma) Posterior Sampler ##

class NeuralDiffusionPosteriorSamplerSigma(nn.Module):
    """
    σ-native NFDM sampler/trainer wrapper.
    """
    def __init__(self, X_dim, theta_dim, n_summaries, num_hidden_layer, device,
                 use_encoder, data_type="iid",
                 sigma_min=0.002, sigma_max=80, sigma_data=0.5,
                 lognormal_loc=-1.2, lognormal_scale=1.2, stats_collector=None):
        super().__init__()
        self.X_dim = X_dim
        self.theta_dim = theta_dim
        self.use_encoder = use_encoder
        self.n_summaries = n_summaries if use_encoder else X_dim
        self.name = "NeuralDiffusionSigma"


        if self.use_encoder:
            if data_type == "iid":
                self.summary = DeepSetSummary(X_dim, n_summaries).to(device)
            elif data_type == "time":
                self.summary = BayesFlowEncoder(X_dim, n_summaries).to(device)
            elif data_type == "set":
                num_head, num_seed = 4, 4
                self.summary = SetEmbedderClean(X_dim, n_summaries, num_head, num_seed).to(device)
            else:
                raise ImportError("Other summary is not supported")
        else:
            self.summary = None

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.lognormal_loc = float(lognormal_loc)
        self.lognormal_scale = float(lognormal_scale)

        self.forward_flow = AffineOT(sigma_min=sigma_min, sigma_max=sigma_max).to(device)
        self.transform = TransformOT(self.forward_flow).to(device)

        self.volatility = VolatilityNeural().to(device)

        self.predictor = Predictor(
            theta_dim=theta_dim,
            hidden_dim=256,
            time_embed_dim=16,
            cond_dim=self.n_summaries,
            cond_mask_prob=0.0,
            num_hidden_layers=num_hidden_layer,
            sigma_data=sigma_data,
            cond_conditional=True,
            device=device
        ).to(device)

        self.nfdm = NeuralDiffusion(self.transform, self.predictor, self.volatility, stats_collector)

        self.device = device


    def loss(self, theta: torch.Tensor, X: torch.Tensor, collect_stats: bool = False, stats_kwargs: dict = None) -> torch.Tensor:

        s = self.summary(X) if self.use_encoder and (self.summary is not None) else X 
        B = theta.shape[0]

        sigma = rand_log_normal(shape=(B,), device=theta.device,
                                loc=self.lognormal_loc, scale=self.lognormal_scale)


        return self.nfdm(theta, sigma, s, collect_stats=collect_stats, stats_kwargs=stats_kwargs).mean()

    @torch.no_grad()
    def sample(self, X: torch.Tensor, num_steps=18, stochastic=False, rho=7.0, sigma_min=None, sigma_max=None,use_heun=False):
        s = self.summary(X) if self.use_encoder and (self.summary is not None) else X
        return self.sample_given_s(s, num_steps, stochastic, rho, sigma_min, sigma_max, use_heun)

    @torch.no_grad()
    def sample_given_s(self, s: torch.Tensor, num_steps=18, stochastic=False, rho=7.0, sigma_min=0.002, sigma_max=80.0, use_heun=False):
        B = s.shape[0]
        sigma_min = max(self.sigma_min, sigma_min or self.sigma_min)
        sigma_max = min(self.sigma_max, sigma_max or self.sigma_max)

        z = torch.randn(B, self.theta_dim, device=s.device) * sigma_max
        
        sig_steps = karras_sigma_grid(num_steps, sigma_min, sigma_max, rho=rho, device=s.device)

        def drift_and_diff(z_t: torch.Tensor, sigma_t: torch.Tensor):

            theta_hat = self.predictor(z_t, sigma_t, s)

            _, dz_dsig, score = self.transform.inverse(z_t, sigma_t, theta_hat, s)

            g = self.volatility(sigma_t)
            f_sigma = dz_dsig - 0.5 * g**2 * score 
            if stochastic:
                return f_sigma, g
            else:
                return f_sigma, z_t.new_zeros(z_t.shape)

        for i, (sig_cur, sig_next) in enumerate(zip(sig_steps[:-1], sig_steps[1:])):
            sig_cur  = sig_cur.expand(B, 1)
            sig_next = sig_next.expand(B, 1)
            dsigma = (sig_next - sig_cur)
            dsigma_sqrt = dsigma.abs().sqrt()

            f_cur, g_cur = drift_and_diff(z, sig_cur)
            w = torch.randn_like(z)
            z_eul = z + dsigma * f_cur + dsigma_sqrt * g_cur * w  # g_cur=0 in deterministic mode

            if use_heun and i < num_steps - 1:
                f_nxt, g_nxt = drift_and_diff(z_eul, sig_next)
                z = z + dsigma * 0.5 * (f_cur + f_nxt) + dsigma_sqrt * 0.5 * (g_cur + g_nxt) * w
            else:
                z = z_eul
        return z



## NFDM (t) Posterior Sampler ##

class NeuralDiffusionPosteriorSampler(torch.nn.Module):
    def __init__(self, X_dim, theta_dim, n_summaries, num_hidden_layer, device, use_encoder, data_type="iid", 
    delta=1e-2, stats_collector=None):
        super().__init__()
        self.X_dim = X_dim
        self.theta_dim = theta_dim
        self.use_encoder = use_encoder
        self.n_summaries = n_summaries if use_encoder else X_dim
        self.name = "NeuralDiffusion"

        # Optionally use a summary network for y
        if self.use_encoder:
            if data_type == "iid":
                self.summary = DeepSetSummary(X_dim, n_summaries).to(device)
            elif data_type == "time":
                self.summary = BayesFlowEncoder(X_dim, n_summaries).to(device)
            elif data_type == "set":
                num_head = 4
                num_seed = 4
                self.summary = SetEmbedderClean(X_dim, n_summaries, num_head, num_seed).to(device)
            else:
                raise ImportError("Other summary is not supported")
        else:
            self.summary = None

        self.forward_flow = AffineOT(self.theta_dim, self.n_summaries, delta=delta).to(device)
        self.transform = AffineTransform(self.forward_flow).to(device)
        self.volatility = VolatilityNeural().to(device)  

        self.predictor = Predictor(
            theta_dim=theta_dim,
            hidden_dim=256,
            time_embed_dim=16,
            cond_dim=self.n_summaries,
            cond_mask_prob=0.0,
            num_hidden_layers=num_hidden_layer,
            device=device,
            cond_conditional=True
        ).to(device)

        self.nfdm = NeuralDiffusion(self.transform, self.predictor, self.volatility)

    def loss(self, theta: torch.Tensor, X: torch.Tensor, collect_stats: bool = False, stats_kwargs: dict = None) -> torch.Tensor:
        s = self.summary(X) if self.use_encoder else X
        t = torch.rand(theta.shape[0], 1, device=theta.device)
        return self.nfdm(theta, t, s, collect_stats=collect_stats, stats_kwargs=stats_kwargs).mean()

    @torch.no_grad()
    def sample(self, X, num_steps=18, stochastic=False, use_heun=True):
        s = self.summary(X) if self.use_encoder else X
        return self.sample_given_s(s, num_steps, stochastic, use_heun)


    @torch.no_grad()
    def sample_given_s(self, s, num_steps=18, stochastic=True, use_heun=True):

        z = torch.randn(s.shape[0], self.theta_dim, device=s.device) # sample z1 from N(0,I)
        
        def sde_drift_and_diffusion(z_t, t):
            t_tensor = t.expand(z_t.shape[0], 1) 
            
            theta_hat = self.predictor(z_t, t_tensor, s) 
            _, dz, score = self.transform.inverse(z_t, t_tensor, theta_hat, s) 
            
            g = self.volatility(t_tensor)
            g2 = g ** 2
            
            drift = dz - 0.5 * g2 * score 
            
            return drift, g
        
        if stochastic:
            final_z, (t_steps, trajectory) = solve_sde(sde_drift_and_diffusion, z, ts=1.0, tf=0.0, n_steps=num_steps, use_heun=use_heun)
        else:
            def ode_drift(z_t, t):
                t_tensor = t.expand(z_t.shape[0], 1)
                theta_hat = self.predictor(z_t, t_tensor, s)
                _, dz, _ = self.transform.inverse(z_t, t_tensor, theta_hat, s)
                return dz
            final_z, (t_steps, trajectory) = solve_ode(ode_drift, z, ts=1.0, tf=0.0, n_steps=num_steps, use_heun=use_heun)

        return final_z

## EDM Posterior Sampler

class DiffusionPosteriorSampler(torch.nn.Module):
    def __init__(self, X_dim, theta_dim, n_summaries,
                 num_hidden_layer,device,use_encoder, data_type="iid", sigma_data=0.5):
        super().__init__()
        self.X_dim = X_dim
        self.theta_dim = theta_dim
        self.use_encoder = use_encoder
        self.n_summaries = n_summaries if use_encoder else X_dim

        if self.use_encoder:
            if data_type == "iid":
                self.summary = DeepSetSummary(X_dim, n_summaries).to(device)
                print("Encoder is for iid data. If not, please check it.")
            elif data_type == "time":
                self.summary = BayesFlowEncoder(X_dim, n_summaries).to(device)
                print("Encoder is for time dependent data. If not, please check it.")
            elif data_type == "set":
                num_head = 4
                num_seed = 4
                self.summary = SetEmbedderClean(X_dim, n_summaries, num_head, num_seed).to(device)
                print("Encoder is for time dependent data. If not, please check it.")
            else:
                raise ImportError("Other summary is not supported")
        else:
            pass

        self.decoder = ScoreNetwork(
            theta_dim=theta_dim,
            hidden_dim=256,
            time_embed_dim=16,
            cond_dim=self.n_summaries,
            cond_mask_prob=0.0,
            num_hidden_layers=num_hidden_layer,
            output_dim=theta_dim,
            device=device,
            cond_conditional=True).to(device)

        self.diffusion = KarrasSDE(theta_dim=theta_dim, data_dim=self.n_summaries, device=device, sigma_data=sigma_data)

    @torch.no_grad()
    def sample(self, X, num_steps=18):
        with torch.no_grad():
            s = self.summary(X) if self.use_encoder else X
            # z, log_p,_ = self.diffusion.sample(self.decoder,s,num_steps)
            z = self.diffusion.edm_sampler(self.decoder, s, num_steps=num_steps)
        return z

    @torch.no_grad()
    def sample_given_s(self,s, num_steps=18):
        z = self.diffusion.edm_sampler(self.decoder, s, num_steps=num_steps)
        return z

    def loss(self, theta, X):
        s = self.summary(X) if self.use_encoder else X
        # diffusion_loss = self.diffusion.diffusion_loss(self.decoder, x, s).mean()
        diffusion_loss = self.diffusion.diffusion_train_step(self.decoder, theta, s)
        return diffusion_loss


## Normalizing Flow Posterior Sampler

class NormalizingFlowPosteriorSampler(torch.nn.Module):
    def __init__(self, X_dim, theta_dim, n_summaries,
                 hidden_dim_decoder, n_flows_decoder,alpha,device,use_encoder,data_type="iid"):
        super().__init__()
        self.X_dim = X_dim
        self.theta_dim = theta_dim
        self.use_encoder = use_encoder
        self.n_summaries = n_summaries if self.use_encoder else X_dim

        if self.use_encoder:
            if data_type == "iid":
                self.summary = DeepSetSummary(X_dim, n_summaries).to(device)
                print("Encoder is for iid data. If not, please check it.")
            elif data_type == "time":
                self.summary = BayesFlowEncoder(X_dim, n_summaries).to(device)
                print("Encoder is for time dependent data. If not, please check it.")
            elif data_type == "set":
                num_head = 4
                num_seed = 4
                self.summary = SetEmbedderClean(X_dim, n_summaries, num_head, num_seed).to(device)
                print("Encoder is for time dependent data. If not, please check it.")
            else:
                raise ImportError("Other summary is not supported")
        else:
            pass

        self.decoder = ConditionalNormalizingFlow(x_dim=theta_dim, y_dim=self.n_summaries, n_cond_layers=3,
                                                         hidden_dim=hidden_dim_decoder, n_params=2,
                                                         n_flows=n_flows_decoder,
                                                         alpha=alpha)

    def forward(self, theta, X):
        # first dimension of X is number of samples in the batch
        # second dimension is number of data points per sample

        # get summary statistics
        s = self.summary(X) if self.use_encoder else X
        z, sum_log_abs_det = self.decoder(theta=theta, X=s)
        return s, z, sum_log_abs_det

    def backward(self, z, X):
        with torch.no_grad():
            s = self.summary(X) if self.use_encoder else X
            theta = self.decoder.backward(z=z, X=s)
        return theta

    @torch.no_grad()
    def sample(self, X):
        with torch.no_grad():
            s = self.summary(X) if self.use_encoder else X
            z = self.decoder.sample(s)
        return z

    def loss(self, theta, X):
        s, z, sum_log_abs_det = self(theta=theta, X=X)
        return 0.5 * (z * z).sum(dim=1) - sum_log_abs_det

    @torch.no_grad()
    def sample_given_s(self, s):
        with torch.no_grad():
            z = self.decoder.sample(s)
        return z
