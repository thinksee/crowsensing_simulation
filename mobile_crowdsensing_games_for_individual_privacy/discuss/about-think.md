1.参考文献REAP:an efficient incentive mechanism ...这篇文章提供的仅是用户的效益和数据的错误率。

2.进一步提升，平台首先首先发布自己的预算，然后使用乘以数据的聚合精度来进行计算平台的效益；
    在用户方面，使用支付-消耗的模式，其中与消耗相关的参数：差分隐私的基本参数等其他。
    
3.差分隐私用在数据上，如何获取较为真实的值从数据上。主要的推导公式和基本的计算步骤。

4.差分隐私中参数使用过大，相当于对整体的数据集是一个异常点或者其他的，这样来讲，差分隐私就显得没有意义。

5.若是只是统计直方图等简单的统计信息的话，使用差分隐私可以对比出相应的效果展示，但是若是是统计其他的信息，如隐私和收益之间的关系，这会产生一个奇特的解释：
    用户勤勤恳恳收集好较优的数据，但是由于隐私偏好，对于个人信息过于保密，导致之前的工作毁于一旦，这样对于激励用户来讲没有太大的意义。
    
6.在服务器端的效用公式，效用 = ? - 支付，其中?是不可以预知的：其中聚合精度是可以预见的，其他的不知道。
    (1)采用预算的方式，但是预算的范围以及如何获取这是一个问题；
    (2)预算也是使用相应的公式，与参与人数->保证了数据的密集程度、个人的信誉值->保证了数据的真实性和可靠性；
    
7.在客户端的效用公式，效用 = 支付 - ?，其中?是如何衡量的：?和差分隐私的参数相关
    (1)隐私的消耗和差分隐私参数和数据量等参数有关。
    (2)
    
8.支付公式的参数相关性
    (1)文章中的参数是按照合约理论进行推导出来的，并不是直接赋值；
    (2)若考虑直接赋值，赋值的直接依据是什么；
    
9.模型
    (0)一个序列{(隐私0,支付0),(隐私1,支付1),(隐私2,支付2),...,(隐私n,支付n)} 
       隐私和支付的关系，数据质量的评定(借助用户信誉，使用百分百的形式)
       设定敏感度阈值，为了防止过大噪声数据对整体数据分布的影响。
       敏感度值-隐私保护水平的发展趋势

    (1)用户效用 = 支付 - 消耗[包括对数据的处理、数据的收集等过程中的耗损]
    (2)服务器效用 = 预算*(精准度) - sum(支付)
    两个问题:支付和隐私等级的关系设计依据;预算的设计依据.
    
10.模型能处理的问题，就是拟合聚合数据的直方图问题。
    (1)对比原始数据和处理数据的拟合程度；
    
11.究其最终的原因，为了解决隐私问题，使用的一些手段。


下一步:
 找:用户信誉和数据质量评定之间的关系论文
    隐私和数据质量评定之间的关系论文
 理:差分隐私在单个数据上的具体的操作步骤，给定数据展示，然后对比隐私加入对数据的影响
    差分隐私在连续数据上的具体操作步骤，给定数据序列，然后对比隐私加入对数据的影响

个性化隐私安全需求的位置隐私保护算法。
首先，根据用户的历史移动轨迹，挖掘用户对不同位置的访问时长、访问频率以及访问的规律性来预测位置对用户的社会属性；
然后，结合位置的自然属性，预测用户-位置的敏感等级；
最后，结合用户在不同的位置有不同的隐私安全需求的特点，设置动态的隐私判定方案，在每个位置选敏感度低的用户参与感知任务。
在确保用户隐私安全的前提下，贡献时空相关性精确高的感知数据。    


12.隐私预算分配上界的证明，首先需要找到相应的评估准则。

13.隐私效果的评判标准