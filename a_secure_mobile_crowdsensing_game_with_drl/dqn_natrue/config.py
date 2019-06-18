__author__ = 'alibaba'
__date__ = '2019/1/2'


class AgentConfig(object):
    scale = 10
    display = False

    max_step = 5 * scale
    memory_size = 10 * scale  # 100

    batch_size = 32  # 神经网络分批处理数据，batch size就是每批处理的样本的个数。
    random_start = 30
    """
    data_format 默认值为"NHWC"，也可以手动设置NCHW，
      这个参数规定了input Tensor和output Tensor的排列方式。排列顺序为[batch, height, width, channels];
    data_format 设置"NCHW" 排列顺序为[batch, channels, height, width]
    """
    cnn_format = 'NCHW'  # GPU上训练使用NCHW格式，在CPU上使用NHWC格式
    discount = 0.99  # 打折参数，防止出现局部最优解
    # 关于学习速率的调整，可参考2015年Cyclical Learning Rates for Training Neural Networks 3.3节
    """
    其中主要思想是从一个低学习率开始训练网络，并在每个batch中指数提高学习率，
    
    """
    target_q_update_step = 1 * scale
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5 * scale

    ep_end = 0.1
    ep_start = 1.
    ep_end_t = memory_size

    history_length = 4
    train_frequency = 4
    learn_start = 5. * scale

    min_delta = -1
    max_delta = 1
    """
    deep Q 是多层神经网络，给定状态s输出一个行动值得向量Q(s,.;θ)，其中θ是神经网络参数。
      对一个n-dim状态空间和一个包含m-dim动作得动作空间，该神经网络是从R^n->R^m得映射。
      DQN算法的两个重要的特点是目标网络(target network)和经验回顾(experience replay).
      目标网络：其参数是θ^-，其实除了其参数每t次从在线网络复制外都和在线网络相同，所以θ^-t = θ-t，其他步都是固定大小。
      经验回顾：观察到的转换被存在放一段时间，并会均匀地从记忆库中采样来更新网络。
      对于上面两种都能大幅度提升算法的性能。
      
    double Q learning，在标准的Q-learing和DQN中的max操作使用同样的值来进行选择和衡量一个行动。
      这实际上更可能选择过高的估计值，从而导致过于乐观的估计值。为了避免着这种情况的出现，我们可以对选择和衡量进行解耦。
      最初的double Q-learing算法中，两个值函数通过将每个经验随机更新两个值函数中的一个，这样就出现了两个权重集合，θ和θ’。
      对每个更新，一个权重集合来确定贪心策略，另外一个用来确定值。
      为了更好地比较两者，我们可以将Q-learning中的选择和衡量分解，
      Yt(Q) = Rt+1 + rQ(St+1, argmax Q(St+1,a;θt);θt)
    """
    double_q = False
    """
    Dueling-DQN:在许多基于视觉的感知的DRL任务中，不同的状态动作对的值函数是不同的，但是在某些状态下，值函数的大小与动作无关。
      此方法使用的是，通过优化神经网络的结构来优化算法。Dueling DQN考虑将Q网络分成两部分，第一部分是仅仅和状态S有关，与具体采用的动作A无关，这部分称之为价值函数V(S,w,a),
      第二部分与状态S和动作A有关，这部分称之为优势函数，A(S,A,w,Bate)，这样一来，价值函数表示为:
          Q(S,A,w,a,bate) = V(S,w,a) + A(S,A,w,bate),其中，w是公共部分的网络参数，而a是价值函数都有部分的网络参数，而b是优势函数都有部分的网络参数。
    Q(S,A,w,a,bate) = V(S,w,a) +(A(S,A,w,bate) - 1/A*sum(A(S,a',w,bate)))
    """
    dueling = False

    _test_step = 5 * scale
    _save_step = _test_step * 10


class EnvironmentConfig(object):
    env_name = 'Breakout-v0'
    screen_width = 84
    screen_height = 84
    max_reward = 1.
    min_reward = -1.


class DQNConfig(AgentConfig, EnvironmentConfig):
    model = ''
    pass


class M1(DQNConfig):
    backend = 'tf'
    env_type = 'detail'
    action_repeat = 1


class M2(DQNConfig):
    pass


def get_config(FLAGS):
    if FLAGS.model == 'm1':
        config = M1
    elif FLAGS.model == 'm2':
        config = M2

    for k, v in FLAGS.flag_values_dict().items():
        if k == 'gpu':
            if v == False:
                config.cnn_format = 'NHWC'
            else:
                config.cnn_format = 'NCHW'

        if hasattr(config, k):
            setattr(config, k, v)

    return config
