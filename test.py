def maddpg(n_episodes=2000, max_t=1000):
    '''
    -------------------------------------------
    Parameters
    
    n_episodes: # of episodes that the agent is training for
    max_t:      # of time steps (max) the agent is taking per episode
    -------------------------------------------
    '''
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]             # turn on train mode of the environment
        states = env_info.vector_observations                         # get the current state for each agent
        maddpg.reset()                                                # reset the OU noise parameter 
        ep_scores = np.zeros(num_agents)                              # initialize the score for each agent
        for t in range(max_t):
            actions = maddpg.act(states)                              # select an action for each agent 
            env_info = env.step(actions)[brain_name]                  # send all actions to the environment
            next_states = env_info.vector_observations                # get next state for each agent
            rewards = env_info.rewards                                # get reward for each agent
            dones = env_info.local_done                               # check if episode finished
            maddpg.step(states, actions, rewards, next_states, dones) # agents record enviroment response in recent step
            states = next_states                                      # set the state as the next state for the following step for each agent
            ep_scores += rewards                                      # update the total score
            if np.any(dones):                                         # exit loop if episode for any agent finished
                break 
                
        scores_deque.append(np.max(ep_scores))
        scores.append(ep_scores)
        
        # print average epsiode score and average 100-episode score for each episode
        print('\rEpisode {} \tMax Score: {:.2f} \tAverage Max Score: {:.2f}'.format(i_episode, np.max(ep_scores), np.mean(scores_deque)), end="")  
        
        # print and save actor and critic weights when a score of +30 over 100 episodes has been achieved
        if np.mean(scores_deque) >= 0.5:
            for i in range(config.num_agents):
                torch.save(maddpg.maddpg_agents[i].actor_local.state_dict(), 'checkpoint_actor_{}_final.pth'.format(i))
                torch.save(maddpg.maddpg_agents[i].critic_local.state_dict(), 'checkpoint_critic_{}_final.pth'.format(i))
            print('\nEnvironment solved in {:d} episodes!\tAverage Max Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            break
    return scores

scores = ddpg_multi()