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


Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
    44 11000.648 MiB 78852.766 MiB          10       @profile
    45                                             def update_stats(self, **stats):
    46 11000.648 MiB    0.000 MiB          10           with self.writer.as_default():
    47 11000.648 MiB    0.000 MiB          50               for key, value in stats.items():
    48 11000.648 MiB    0.000 MiB          40                   tf.summary.scalar(key, value, step=self.step)
    49 11000.648 MiB    0.000 MiB          10               self.writer.flush()
    50 11000.648 MiB    0.000 MiB          10           self.step += 1


Filename: /home/lorin/Documents/carla/PythonAPI/examples/dqn_agent.py

Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
    69  660.184 MiB  565.031 MiB           2       @profile
    70                                             def create_model(self):
    71  662.184 MiB   96.848 MiB           2           base_model = Xception(weights=None, include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    72  662.184 MiB    0.000 MiB           2           x = base_model.output
    73  662.184 MiB    0.000 MiB           2           x = GlobalAveragePooling2D()(x)
    74                                         
    75                                                 # This is a 3-neuron output representing the 3 actions the agent could take
    76  662.434 MiB    0.555 MiB           2           predictions = Dense(3, activation="linear")(x)
    77                                         
    78  662.434 MiB    0.000 MiB           2           model = Model(inputs=base_model.input, outputs=predictions)
    79  662.434 MiB    0.000 MiB           2           model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
    80  662.434 MiB    0.000 MiB           2           return model


Filename: /home/lorin/Documents/carla/PythonAPI/examples/dqn_agent.py

Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
   132  762.965 MiB  762.965 MiB           1       @profile
   133                                             def train_in_loop(self):
   134  773.504 MiB   10.539 MiB           1           X = np.random.uniform(size=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(np.float32)
   135  773.504 MiB    0.000 MiB           1           y = np.random.uniform(size=(1, 3)).astype(np.float32)
   136                                         
   137 1467.551 MiB  694.047 MiB           1           self.model.fit(X,y, verbose=False, batch_size=1)
   138                                         
   139 1467.551 MiB    0.000 MiB           1           self.training_initialized = True
   140                                         
   141 1467.551 MiB    0.000 MiB           1           while True:
   142 3423.441 MiB   16.781 MiB        2170               if self.terminate:
   143                                                         return
   144 3864.137 MiB  459.242 MiB        2170               self.train()
   145 3423.441 MiB 1912.328 MiB        2169               time.sleep(0.01)


Filename: /home/lorin/Documents/carla/PythonAPI/examples/vehicle_environment.py

Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
    58 10939.340 MiB 750765.727 MiB          98       @profile
    59                                             def spawn_vehicle_and_agent(self):
    60                                                 """
    61                                                 Repeatedly attempts to spawn a vehicle and actor in the world. If an exception
    62                                                 is thrown generating the spawn point (e.g. due to collision), try again.
    63                                                 """
    64 10939.340 MiB  -56.871 MiB          98           spawn_successful = False
    65 10939.340 MiB -113.742 MiB         196           while not spawn_successful:
    66 10939.340 MiB  -56.871 MiB          98               try:
    67 10939.340 MiB  -56.871 MiB          98                   spawn_point = random.choice(self.world.get_map().get_spawn_points())
    68 10939.340 MiB  -56.871 MiB          98                   self.vehicle = self.world.spawn_actor(self.mini_cooper, spawn_point)
    69 10939.340 MiB  -56.871 MiB          98                   self.actor_list.append(self.vehicle)
    70 10939.340 MiB  -56.871 MiB          98                   spawn_successful = True
    71                                                     except Exception as e:
    72                                                         print(e)


Filename: /home/lorin/Documents/carla/PythonAPI/examples/vehicle_environment.py

Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
   143 11000.773 MiB 697625.539 MiB       62760       @profile
   144                                             def process_image(self, image):
   145                                                 """
   146                                                 CARLA image data from the camera sensor is RGB-alpha. We'll reshape the raw
   147                                                 data, then extract just the RGB channels, ignoring the alpha data.
   148                                                 """
   149 11000.773 MiB -57666.012 MiB       62760           raw_image_data = np.array(image.raw_data)
   150 11000.773 MiB -59610.078 MiB       62760           reshaped_image = raw_image_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
   151 11000.773 MiB -59619.086 MiB       62760           rgb_extracted_image = reshaped_image[:, :, :3]
   152 11000.773 MiB -59615.383 MiB       62760           if DISPLAY_PREVIEW:
   153                                                     cv2.imshow("", rgb_extracted_image)
   154                                                     cv2.waitKey(1)
   155 11000.773 MiB -59608.453 MiB       62760           self.front_camera = rgb_extracted_image