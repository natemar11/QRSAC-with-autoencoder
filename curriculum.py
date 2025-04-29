class SimpleEpochCurriculum:
    """Curriculum with 150 epochs per stage"""
    
    def __init__(self, epochs_per_stage=150, enabled=True):
        self.enabled = enabled
        self.current_epoch = 0
        self.epochs_per_stage = epochs_per_stage
        self.total_stages = 3
        
        # Stage descriptions for logging
        self.stage_descriptions = {
            0: "BEGINNER STAGE: Focus on basic track following with no reward for speed",
            1: "INTERMEDIATE STAGE: Introducing speed while maintaining track position",
            2: "ADVANCED STAGE: Optimizing for speed"
        }
        
        # Define curriculum parameters per stage
        self.curriculum_params = {
            # Stage 0: Beginner
            0: {
                "speed_reward_weight": 0.2,
                "max_cte_error": 4.5,
                "track_width_factor": 1.2
            },
            # Stage 1: Intermediate
            1: {
                "speed_reward_weight": 0.3,
                "max_cte_error": 3.0,
                "track_width_factor": 1.0
            },
            # Stage 2: Final
            2: {
                "speed_reward_weight": 0.5,
                "max_cte_error": 2.0,
                "track_width_factor": 0.8
            }
        }
        
        # Print initial stage info
        if enabled:
            self._print_stage_info(0)
    
    def _print_stage_info(self, stage):
        """Print detailed information about the current stage"""
        params = self.curriculum_params[stage]
        print("\n" + "="*80)
        print(f"CURRICULUM STAGE {stage}: {self.stage_descriptions[stage]}")
        print(f"Current Epoch: {self.current_epoch}")
        print(f"Epochs in this stage: {self.current_epoch} - {self.current_epoch + self.epochs_per_stage - 1}")
        print(f"Parameters for this stage:")
        print(f"  - Max CTE Error: {params['max_cte_error']}")
        print(f"  - Track Width Factor: {params['track_width_factor']}")
        print("="*80 + "\n")
    
    def get_current_stage(self):
        """Determine current stage based on epoch count"""
        if not self.enabled:
            return self.total_stages - 1  # Always return final stage if disabled
        
        stage = (self.current_epoch // self.epochs_per_stage) % self.total_stages
        return stage
    
    def get_current_params(self):
        """Get parameters for current curriculum stage"""
        return self.curriculum_params[self.get_current_stage()]
    
    def advance_epoch(self):
        """Move to the next epoch and update stage if needed"""
        old_stage = self.get_current_stage()
        self.current_epoch += 1
        new_stage = self.get_current_stage()
        
        # Return True if we moved to a new stage
        if new_stage != old_stage:
            self._print_stage_info(new_stage)
            # Print completion message for previous stage
            print(f"Completed Stage {old_stage} after {self.epochs_per_stage} epochs")
            return True
        return False
    
    def get_reward_function(self):
        """Returns a reward function configured for the current stage"""
        
        def reward_fn(env, done):
            """
            Custom reward function that adapts based on curriculum stage
            """
            # Get current parameters for this stage
            curr_params = self.get_current_params()
            
            # Basic failure conditions (consistent across stages)
            if done:
                return -1.0
            
            if env.hit != "none":
                return -2.0
            
            # Get normalized CTE (cross track error)
            normalized_cte = abs(env.cte) / curr_params["max_cte_error"]
            
            # Base track position reward (common across all stages)
            position_reward = 1.0 - normalized_cte
            
            if not self.enabled:
                # Standard mode: Use final stage parameters
                position_reward*=curr_params["track_width_factor"]
                speed_reward = env.forward_vel *curr_params["speed_reward_weight"]
                return position_reward + speed_reward 
            
            # Stage-specific rewards
            stage = self.get_current_stage()
            
            if stage == 0:
                # Stage 1: Focus on staying on track
                return position_reward
            
            elif stage == 1:
                # Stage 2: Start considering speed but with lower weight
                speed_reward = env.forward_vel*curr_params["speed_reward_weight"]
                return position_reward + speed_reward
            
            else:  # stage == 2
                # Stage 3: Full optimization
                track_width_factor = curr_params["track_width_factor"]
                position_reward *= track_width_factor
                speed_reward = env.forward_vel*curr_params["speed_reward_weight"]
                return position_reward + speed_reward
        
        return reward_fn
    
    def apply_to_env(self, env):
        """Apply current curriculum parameters to environment"""
        params = self.get_current_params()
        
        # Set the reward function
        env.set_reward_fn(self.get_reward_function())
        
        # Apply other parameters to DonkeyCar environment
        if hasattr(env, 'unwrapped'):
            if hasattr(env.unwrapped, 'set_max_cte'):
                env.unwrapped.set_max_cte(params["max_cte_error"])
            if hasattr(env.unwrapped, 'set_track_width'):
                env.unwrapped.set_track_width(params["track_width_factor"])
        
        return env
