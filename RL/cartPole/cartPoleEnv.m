classdef cartPoleEnv < rl.env.MATLABEnvironment
    % This code is based on Paulo Carvalho's code (2019)    
    properties
        % Select environment name here 
        open_env = py.gym.make('CartPole-v0'); 
    end
    methods              
        function this = cartPoleEnv()
            % Initialize Observation settings
            ObservationInfo             = rlNumericSpec([4 1]);
            ObservationInfo.Name        = 'cartpole';
            ObservationInfo.Description = 'images';         
            ActionInfo                  = rlFiniteSetSpec([0 1]); 
            ActionInfo.Name             = 'left, right';            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
        end        
        function [observation, reward, done, info] = step(this,Action)
            result      = this.open_env.step(int16(Action));  
            
            observation = single(result{1}); 
            reward      = double(result{2});
            done        = result{3};
            info        = [];                 
        end
        function initialObservation = reset(this)
            result             = this.open_env.reset();
            initialObservation = single(result);
        end
        function screen = render(this)
            screen = this.open_env.render();
        end
        function close(this)
            this.open_env.close()
        end
    end
end
