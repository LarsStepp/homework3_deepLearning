import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import environment


class CNMP(torch.nn.Module):
    def __init__(self, d_x, d_y, d_c, hidden_size, num_hidden_layers, min_std=0.1):
        super(CNMP, self).__init__()
        self.d_x = d_x  # query dimension (time)
        self.d_y = d_y  # target dimension (ey, ez, oy, oz)
        self.d_c = d_c  # condition dimension (height)

        # Encoder: processes context (t, y) pairs
        self.encoder = []
        self.encoder.append(torch.nn.Linear(self.d_x + self.d_y, hidden_size))
        self.encoder.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
        self.encoder = torch.nn.Sequential(*self.encoder)

        # Decoder: predicts targets given context representation + condition + query
        self.decoder = []
        self.decoder.append(torch.nn.Linear(hidden_size + self.d_c + self.d_x, hidden_size))
        self.decoder.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.decoder.append(torch.nn.Linear(hidden_size, hidden_size))
            self.decoder.append(torch.nn.ReLU())
        self.decoder.append(torch.nn.Linear(hidden_size, 2 * self.d_y))
        self.decoder = torch.nn.Sequential(*self.decoder)

        self.min_std = min_std

    def nll_loss(self, observation, target, condition, target_truth, observation_mask=None, target_mask=None):
        '''
        Negative log-likelihood loss for training CNMP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor that contains context points.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor that contains query times.
        condition : torch.Tensor
            (n_batch, d_c) sized tensor with condition (height).
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor with ground truth targets.
        observation_mask : torch.Tensor
            (n_batch, n_context) mask for context points.
        target_mask : torch.Tensor
            (n_batch, n_target) mask for target points.
        '''
        mean, std = self.forward(observation, target, condition, observation_mask)
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        return loss

    def forward(self, observation, target, condition, observation_mask=None):
        '''
        Forward pass of CNMP.
        '''
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate(r, condition, target)
        decoder_out = self.decoder(h_cat)
        mean = decoder_out[..., :self.d_y]
        logstd = decoder_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std

    def encode(self, observation):
        h = self.encoder(observation)
        return h

    def aggregate(self, h, observation_mask):
        if observation_mask is not None:
            h = (h * observation_mask.unsqueeze(2)).sum(dim=1)
            normalizer = observation_mask.sum(dim=1).unsqueeze(1)
            r = h / normalizer
        else:
            r = h.mean(dim=1)
        return r

    def concatenate(self, r, condition, target):
        num_target_points = target.shape[1]
        r = r.unsqueeze(1).repeat(1, num_target_points, 1)
        condition = condition.unsqueeze(1).repeat(1, num_target_points, 1)
        h_cat = torch.cat([r, condition, target], dim=-1)
        return h_cat


class Hw5Env(environment.BaseEnv):
    def __init__(self, render_mode="gui") -> None:
        self._render_mode = render_mode
        self.viewer = None
        self._init_position = [0.0, -np.pi/2, np.pi/2, -2.07, 0, 0, 0]
        self._joint_names = [
            "ur5e/shoulder_pan_joint",
            "ur5e/shoulder_lift_joint",
            "ur5e/elbow_joint",
            "ur5e/wrist_1_joint",
            "ur5e/wrist_2_joint",
            "ur5e/wrist_3_joint",
            "ur5e/robotiq_2f85/right_driver_joint"
        ]
        self.reset()
        self._joint_qpos_idxs = [self.model.joint(x).qposadr for x in self._joint_names]
        self._ee_site = "ur5e/robotiq_2f85/gripper_site"

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [0.5, 0.0, 1.5]
        height = np.random.uniform(0.03, 0.1)
        self.obj_height = height
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, height], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="frontface")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=0).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[1:]
        obj_pos = self.data.body("obj1").xpos[1:]
        return np.concatenate([ee_pos, obj_pos, [self.obj_height]])


def bezier(p, steps=100):
    t = np.linspace(0, 1, steps).reshape(-1, 1)
    curve = np.power(1-t, 3)*p[0] + 3*np.power(1-t, 2)*t*p[1] + 3*(1-t)*np.power(t, 2)*p[2] + np.power(t, 3)*p[3]
    return curve


if __name__ == "__main__":
    # Step 1: Collect demonstrations
    env = Hw5Env(render_mode="offscreen")
    trajectories = []
    
    for i in range(100):
        env.reset()
        p_1 = np.array([0.5, 0.3, 1.04])
        p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
        p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p_4 = np.array([0.5, -0.3, 1.04])
        points = np.stack([p_1, p_2, p_3, p_4], axis=0)
        curve = bezier(points)

        env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
        states = []
        times = np.linspace(0, 1, len(curve))
        for j, p in enumerate(curve):
            env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
            state = env.high_level_state()  # (ey, ez, oy, oz, h)
            states.append(state)
        
        states = np.stack(states)  # (T, 5)
        # Extract: time, ey, ez, oy, oz, h
        trajectory = {
            't': times,  # (T,)
            'ey': states[:, 0],  # (T,)
            'ez': states[:, 1],  # (T,)
            'oy': states[:, 2],  # (T,)
            'oz': states[:, 3],  # (T,)
            'h': states[0, 4]  # scalar
        }
        trajectories.append(trajectory)
        print(f"Collected {i+1} trajectories.", end="\r")
    
    print("\nTraining CNMP model...")
    
    # Step 2: Train CNMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNMP(d_x=1, d_y=4, d_c=1, hidden_size=128, num_hidden_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    for epoch in range(100):
        total_loss = 0.0
        for trajectory in trajectories:
            # Create context and target split
            T = len(trajectory['t'])
            n_context = np.random.randint(1, min(51, T))
            n_target = np.random.randint(1, min(51, T))
            
            context_idx = np.random.choice(T, n_context, replace=False)
            target_idx = np.random.choice(T, n_target, replace=False)
            
            # Build tensors
            t_context = torch.tensor(trajectory['t'][context_idx], dtype=torch.float32).view(-1, 1).to(device)
            y_context = torch.tensor(np.stack([
                trajectory['ey'][context_idx],
                trajectory['ez'][context_idx],
                trajectory['oy'][context_idx],
                trajectory['oz'][context_idx]
            ], axis=1), dtype=torch.float32).to(device)
            
            observation = torch.cat([t_context, y_context], dim=1).unsqueeze(0)  # (1, n_context, 5)
            
            t_target = torch.tensor(trajectory['t'][target_idx], dtype=torch.float32).view(-1, 1).to(device)
            y_target = torch.tensor(np.stack([
                trajectory['ey'][target_idx],
                trajectory['ez'][target_idx],
                trajectory['oy'][target_idx],
                trajectory['oz'][target_idx]
            ], axis=1), dtype=torch.float32).to(device)
            
            target = t_target.unsqueeze(0)  # (1, n_target, 1)
            target_truth = y_target.unsqueeze(0)  # (1, n_target, 4)
            
            condition = torch.tensor([[trajectory['h']]], dtype=torch.float32).to(device)  # (1, 1)
            
            loss = model.nll_loss(observation, target, condition, target_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {total_loss/len(trajectories):.6f}")
    
    print("\nRunning 100 tests...")
    
    # Step 3: Test with random observations and queries
    mse_ee_list = []
    mse_obj_list = []
    
    for test_idx in range(100):
        traj_idx = np.random.randint(len(trajectories))
        trajectory = trajectories[traj_idx]
        
        T = len(trajectory['t'])
        n_obs = np.random.randint(1, min(51, T))
        n_query = np.random.randint(1, min(51, T))
        
        obs_idx = np.random.choice(T, n_obs, replace=False)
        query_idx = np.random.choice(T, n_query, replace=False)
        
        # Build observation tensor
        t_obs = torch.tensor(trajectory['t'][obs_idx], dtype=torch.float32).view(-1, 1).to(device)
        y_obs = torch.tensor(np.stack([
            trajectory['ey'][obs_idx],
            trajectory['ez'][obs_idx],
            trajectory['oy'][obs_idx],
            trajectory['oz'][obs_idx]
        ], axis=1), dtype=torch.float32).to(device)
        
        observation = torch.cat([t_obs, y_obs], dim=1).unsqueeze(0)
        
        # Build target tensor
        t_query = torch.tensor(trajectory['t'][query_idx], dtype=torch.float32).view(-1, 1).to(device)
        y_query_truth = torch.tensor(np.stack([
            trajectory['ey'][query_idx],
            trajectory['ez'][query_idx],
            trajectory['oy'][query_idx],
            trajectory['oz'][query_idx]
        ], axis=1), dtype=torch.float32).to(device)
        
        target = t_query.unsqueeze(0)
        y_query_truth = y_query_truth.unsqueeze(0)
        
        condition = torch.tensor([[trajectory['h']]], dtype=torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            mean, _ = model.forward(observation, target, condition)
        
        # Calculate MSE for end-effector (ey, ez) and object (oy, oz)
        mse_ee = ((mean[0, :, :2] - y_query_truth[0, :, :2]) ** 2).mean().item()
        mse_obj = ((mean[0, :, 2:] - y_query_truth[0, :, 2:]) ** 2).mean().item()
        
        mse_ee_list.append(mse_ee)
        mse_obj_list.append(mse_obj)
    
    # Step 4: Plot results
    mean_ee = np.mean(mse_ee_list)
    std_ee = np.std(mse_ee_list)
    mean_obj = np.mean(mse_obj_list)
    std_obj = np.std(mse_obj_list)
    
    print(f"\nEnd-Effector MSE: {mean_ee:.6f} ± {std_ee:.6f}")
    print(f"Object MSE: {mean_obj:.6f} ± {std_obj:.6f}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['End-Effector', 'Object'], [mean_ee, mean_obj], 
                   yerr=[std_ee, std_obj], capsize=10, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('CNMP Prediction Error (100 tests)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("cnmp_results.png")
