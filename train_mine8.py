from collections import deque
import random
import gym
import numpy as np
from matplotlib import pylab
from skimage.feature._cascade import rgb2gray
from skimage.transform import resize
import time
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class RacerAgent():
    def __init__(self,
                 action_space =[
                    (-1.0, +1.0, +0.0), (+1.0, +1.0, +0.0),(+0.0, +1.0, +0.0)
            ]):


        # 화면송출설정
        self.render = False

        # 상태정의
        self.state_size = (96, 96, 4)
        # 행동정의
        self.action_space = action_space

        # DQN 하이퍼파라미터

        # 전반적 학습의 구도를 결정
        self.exploration_steps = 1000000.
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.01
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps

        self.learning_rate = 1e-4

        self.batch_size = 32
        self.train_start = 5000
        self.update_target_rate = 2000
        self.discount_factor = 0.9

        # 리플레이 메모리 (max 20,000)
        self.memory = deque(maxlen=20000)

        # model 과 target_model 생성
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()


        # Action_Space 재정의 Continous -> Discrete
        # [     Steer        /     Gas       /     Brake         ]
        # [    -1 ~ +1       /    0 ~ +1     /     0 ~ +1        ]-> float 형으로 그냥 범위안에서 정의

        # Discrete하게 변형
        # [-1,-0.5,0,+0.5,+1 /    0, +1     /   0, +0.5, +1      ]-> float 형으로 상수로 정의
        # [왼,왼45',중,오45',오/  가속O, 가속X / 관성,약간감속,완전감속 ]

        # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(6, (7, 7), strides=3, activation='relu', input_shape=self.state_size))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(12, (4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate, epsilon=1e-7))
        model.summary()
        return model

    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            action_index = random.randrange(len(self.action_space))
        else:
            q_value = self.model.predict(history)
            action_index = np.argmax(q_value[0])
        return self.action_space[action_index]

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, done):
        self.memory.append((history, action, reward, next_history, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                                self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                     self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward,  = [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])

        target_value = self.target_model.predict(next_history)

        for i in range(self.batch_size):
            if done:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                            np.amax(target_value[i])

        self.model.fit(history, target, batch_size=self.batch_size, epochs=1, verbose=0)

    def save(self, name):
        self.target_model.save_weights(name)

def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (96, 96), mode='constant') * 255)
    return processed_observe

# main loop
if __name__ == "__main__":
    env = gym.make('CarRacing-v0')
    agent = RacerAgent()

    rewards, episodes, global_step = [], [], 0

    EPISODES = 500
    total_reward = 0

    start = time.time()
    for e in range(EPISODES):
        done = False
        step, score = 0, 0
        observe = env.reset() # state 대신 observe
        #print(observe)

        # 프레임 띄어서 4개의 state를 history로 저장
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 96, 96, 4))
        #print(history)
        total_reward = 0

        while not done:
            if agent.render:
                env.render()

            # 바로 전 4개의 상태로 행동을 선택
            action = agent.get_action(history)# action을 먼저 받아오는 과정 step1
            #print(history)
            #print(action)

            reward_sum = 0
            #for _ in range(4):
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, info = env.step(action)
            reward_sum += reward
            #if done:
            #    break

            # 각 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 96, 96, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            step += 1
            global_step += 1

            #Reward acceration
            if action[1] == 1 and action[2] == 0:
                reward_sum *= 1.5

            total_reward += reward_sum


            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, done)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()


            if done:

                rewards.append(total_reward)
                episodes.append(e)
                pylab.title("Performance-Graph")
                pylab.suptitle("case8")
                pylab.xlabel("Episode")
                pylab.ylabel("Total_Reward")
                pylab.plot(episodes, rewards, 'b')
                pylab.savefig("./case8/save_graph/auto_drive_graph.png")
                print("episode:", e, "  reward:", total_reward, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step)
                      #"  average_q:",agent.avg_q_max / float(step), \
                      #"  average loss:",agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0

                print("The time used to execute this is given below")
                end = time.time()
                print(end - start)

        # 200 에피소드마다 모델 저장
        if e % 100 == 0:
            agent.save("./case8/save_model/auto_drive_train_Episode_{}.h5".format(e))
    env.close()




