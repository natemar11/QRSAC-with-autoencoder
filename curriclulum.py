class DonkeyCarCurriculum:
    """Manages curriculum learning for DonkeyCar"""
    
    def __init__(self, 
                 initial_stage=0,
                 max_stages=5,
                 promotion_threshold=0.8,
                 evaluation_window=20):
        self.current_stage = initial_stage
        self.max_stages = max_stages
        self.promotion_threshold = promotion_threshold
        self.evaluation_window = evaluation_window
        
        # Track recent performance
        self.recent_rewards = []
        self.recent_successes = []
        
        # Define curriculum parameters per stage
        self.curriculum_params = {
            # Stage 0: Beginner - Slow speed, wide track
            0: {
                "max_speed": 0.3,
                "throttle_reward_weight": 0.1,
                "steering_penalty_weight": 0.1,
                "max_cte_error": 5.0,
                "track_width_factor": 1.4
            },
            # Stage 1: Easy - Moderate speed, normal track
            1: {
                "max_speed": 0.7,
                "throttle_reward_weight": 0.2,
                "steering_penalty_weight": 0.2,
                "max_cte_error": 4.0,
                "track_width_factor": 1.2
            },
            # Stage 2: Normal - Regular conditions
            2: {
                "max_speed": 1.0,
                "throttle_reward_weight": 0.4,
                "steering_penalty_weight": 0.4,
                "max_cte_error": 3.0,
                "track_width_factor": 1.0
            },
            
        }
    
    def get_current_params(self):
        """Get parameters for current curriculum stage"""
        return self.curriculum_params[min(self.current_stage, self.max_stages-1)]
    
    def update_performance(self, episode_reward, success=None):
        """Track recent performance and update curriculum stage if needed"""
        self.recent_rewards.append(episode_reward)
        if success is not None:
            self.recent_successes.append(float(success))
        
        # Keep only the most recent episodes
        if len(self.recent_rewards) > self.evaluation_window:
            self.recent_rewards = self.recent_rewards[-self.evaluation_window:]
        if len(self.recent_successes) > self.evaluation_window:
            self.recent_successes = self.recent_successes[-self.evaluation_window:]
        
        # Check if we should advance to the next stage
        if len(self.recent_successes) >= self.evaluation_window:
            success_rate = sum(self.recent_successes) / len(self.recent_successes)
            if success_rate >= self.promotion_threshold and self.current_stage < self.max_stages-1:
                self.current_stage += 1
                print(f"🎓 Curriculum advanced to stage {self.current_stage}")
                # Reset performance tracking for new stage
                self.recent_rewards = []
                self.recent_successes = []
                return True
        
        return False
    
    def apply_to_env(self, env):
        """Apply current curriculum parameters to environment"""
        params = self.get_current_params()
        
        # Apply to DonkeyCar environment
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'set_reward_weights'):
            # Assuming these methods exist or can be added to your env
            env.unwrapped.set_max_speed(params["max_speed"])
            env.unwrapped.set_reward_weights(
                throttle_weight=params["throttle_reward_weight"],
                steering_weight=params["steering_penalty_weight"]
            )
            env.unwrapped.set_max_cte(params["max_cte_error"])
            env.unwrapped.set_track_width(params["track_width_factor"])
        
        return env
