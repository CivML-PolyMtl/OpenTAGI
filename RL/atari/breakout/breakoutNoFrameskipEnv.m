classdef breakoutNoFrameskipEnv < rl.env.MATLABEnvironment
    % This code is based on Paulo Carvalho's code (2019)    
    properties
        % Select environment name here 
        open_env = py.gym.make('BreakoutNoFrameskip-v4'); 
    end
    methods              
        function this = breakoutNoFrameskipEnv()
            % Initialize Observation settings
            ObservationInfo             = rlNumericSpec([4 1]);
            ObservationInfo.Name        = 'breakout';
            ObservationInfo.Description = 'images';         
            ActionInfo                  = rlFiniteSetSpec([0:3]); 
            ActionInfo.Name             = ' ';            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
        end        
        function [observation, reward, done, info] = step(this,Action)
            result      = this.open_env.step(int16(Action));           
            observation = uint8(double(result{1})); 
            reward      = single(double(result{2}));
            done        = logical(double(result{3}));
            info        = result{4};
            info        = double(info{'ale.lives'});                 
        end
        function seed(this, s)
            this.open_env.seed(int32(s));  
        end
        function InitialObservation = reset(this)
            result             = this.open_env.reset();
            InitialObservation = single(result);
        end
        function lives = episodicLife(this)
            lives = double(this.open_env.unwrapped.ale.lives());
        end
        function meanings = get_action_meanings(this)
            meanings = this.open_env.unwrapped.get_action_meanings();
        end
        function screen = render(this)
            screen = this.open_env.render('rgb_array');
        end
        function close(this)
            this.open_env.close()
        end
    end
end
