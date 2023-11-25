Memory profiler on the DQNAgent.py training method


Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
    84 3421.758 MiB 3384.164 MiB        2130       @profile
    85                                             def train(self):
    86 3421.758 MiB   16.352 MiB        2130           if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
    87 3420.727 MiB   16.594 MiB        2129               return
    88                                         
    89 3421.758 MiB    0.000 MiB           1           minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
    90                                         
    91 3548.922 MiB  127.164 MiB          19           current_states = np.array([transition[0] for transition in minibatch])/255
    92 3561.113 MiB   12.191 MiB           1           current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)
    93                                         
    94 3677.082 MiB  115.969 MiB          19           new_current_states = np.array([transition[3] for transition in minibatch])/255
    95 3738.156 MiB   61.074 MiB           1           future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)
    96                                         
    97 3738.156 MiB    0.000 MiB           1           X = []
    98 3738.156 MiB    0.000 MiB           1           y = []
    99                                         
   100 3738.156 MiB    0.000 MiB          17           for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
   101 3738.156 MiB    0.000 MiB          16               if not done:
   102 3738.156 MiB    0.000 MiB          16                   max_future_q = np.max(future_qs_list[index])
   103 3738.156 MiB    0.000 MiB          16                   new_q = reward + DISCOUNT * max_future_q
   104                                                     else:
   105                                                         new_q = reward
   106                                         
   107 3738.156 MiB    0.000 MiB          16               current_qs = current_qs_list[index]
   108 3738.156 MiB    0.000 MiB          16               current_qs[action] = new_q
   109                                         
   110 3738.156 MiB    0.000 MiB          16               X.append(current_state)
   111 3738.156 MiB    0.000 MiB          16               y.append(current_qs)
   112                                         
   113 3738.156 MiB    0.000 MiB           1           log_this_step = False
   114 3738.156 MiB    0.000 MiB           1           if self.tensorboard.step > self.last_logged_episode:
   115 3738.156 MiB    0.000 MiB           1               log_this_step = True
   116 3738.156 MiB    0.000 MiB           1               self.last_log_episode = self.tensorboard.step
   117                                         
   118 3859.359 MiB  121.203 MiB           1           self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
   119 3865.676 MiB    6.316 MiB           1                          callbacks=[self.tensorboard] if log_this_step else None)
   120                                         
   121                                                 if log_this_step:
   122                                                     self.target_update_counter += 1
   123                                         
   124                                                 if self.target_update_counter > UPDATE_TARGET_EVERY:
   125                                                     self.target_model.set_weights(self.model.get_weights())
   126                                                     self.target_update_counter = 0
