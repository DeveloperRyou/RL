import tensorflow as tf
import numpy as np

#환경

print('환경 정의')
class contextual_bandit():
    def __init__(self):
        self.state=0
        self.bandits=np.array([[0.2,0,0.1,-5],[0.1,3,-2,1],[25,0,5,2]]) #4,3,2arm이 최적
        self.num_bandit=self.bandits.shape[0] #슬롯머신 개수
        self.num_arms=self.bandits.shape[1] #손잡이 개수

    def get_bandit(self): #슬롯머신 선택
        self.state=np.random.randint(0,self.num_bandit)#랜덤으로 슬롯머신 선택
        return self.state

    def pullArm(self,action):
        #state와 action으로 손잡이 당김
        bandit=self.bandits[self.state,action]
        
        #표준편차 1의 정규분포에서 랜덤하게 난수생성
        result=np.random.randn(1)
    
        if result>bandit: #양의 보상 반환
            return 1
        else: #음의 보상 반환
            return -1

#에이전트

class agent():
    def __init__(self,learning_rate,num_bandit,num_arm):
        #현재 환경 입력(슬롯머신 index)
        self.state_in=tf.placeholder(tf.int32,[1])
        #환경을 one_hot encoding으로 표현
        state_in_onehot=tf.onehot(self.state_in,num_bandit)
        


def pullBandit(bandit):
    #표준편차 1의 정규분포에서 랜덤하게 난수생성
    result=np.random.randn(1)
    
    if result>bandit: #양의 보상 반환
        return 1
    else: #음의 보상 반환
        return -1

#에이전트

print('에이전트 정의')
reward_holder=tf.placeholder(tf.float32,[1]) #선택한 arm의 보상
action_holder=tf.placeholder(tf.int32,[1]) #선택할 arm

W=tf.Variable(tf.ones([num_arms])) #가중치
output=tf.nn.softmax(W) #W에 의해 각각의 arm을 선택할 확률
Real_output=tf.slice(output,action_holder,[1]) #선택된 arm의 output
loss=-(tf.log(Real_output[0])*reward_holder[0])

optimizer=tf.train.AdamOptimizer(learning_rate=0.001)
train_step=optimizer.minimize(loss) #학습

#세션 설정

print('학습 전처리')
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#텐서보드
loss_hist=tf.summary.scalar('loss',loss)
weight_hist=tf.summary.histogram('weight_softmax',output)
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('./board/Multiarmed',sess.graph)

#학습

print('학습시작')
total_episodes=2000 #학습횟수
total_reward=np.zeros([num_arms]) #총 보상

for i in range(total_episodes):     
   
    #볼츠만 분포로 arm 선택
    actions=sess.run(output) #arm을 선택할 확률
    action=np.random.choice(num_arms,p=actions) #실제 선택한 arm
    
    reward=pullBandit(bandit_arms[action]) #arm을 당긴다

    #학습
    sess.run(train_step,feed_dict={reward_holder:[reward],action_holder:[action]})

    #총보상 업데이트
    total_reward[action]+=reward 
    if i%100==0:
        print(total_reward)
    
    #텐서보드 
    summary=sess.run(merged,feed_dict={reward_holder:[reward],action_holder:[action]})
    writer.add_summary(summary,i)
    writer.flush()
