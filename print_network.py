import torch
from models.r3gan_networks import Generator, Discriminator

# 参数设置
WidthPerStage = [3 * x // 4 for x in [1024, 1024, 1024, 1024]] # 每个ResidualBlock的输入与下个的输出通道数
BlocksPerStage = [2 * x for x in [1, 1, 1, 1]] # 每个GeneratorBlock的ResidualBlock的个数
CardinalityPerStage = [3 * x for x in [32, 32, 32, 32]] # 每个ResidualBlock的中间分组卷积层的组数
NoiseDimension = 3 # 张量输入维度
# ConditionEmbeddingDimension = 64  # 假设有条件

# 构建生成器
netG = Generator(
    NoiseDimension=NoiseDimension,
    WidthPerStage=WidthPerStage,
    CardinalityPerStage=CardinalityPerStage,
    BlocksPerStage=BlocksPerStage,
    ExpansionFactor=2, # 每个ResidualBlock的卷积输出通道的扩展因子
    # ConditionDimension=ConditionEmbeddingDimension,
    # ConditionEmbeddingDimension=ConditionEmbeddingDimension,
    KernelSize=3, # 分组卷积层卷积核大小
    ResamplingFilter=None
)

# 构建判别器
netD = Discriminator(
    WidthPerStage=WidthPerStage,
    CardinalityPerStage=CardinalityPerStage,
    BlocksPerStage=BlocksPerStage,
    ExpansionFactor=2,
    # ConditionDimension=ConditionEmbeddingDimension,
    # ConditionEmbeddingDimension=WidthPerStage[0],
    KernelSize=3,
    ResamplingFilter=None
    # ResamplingFilter=[1, 1, 1]
)

# 打印网络结构
print('Generator:')
print(netG)
print('\nDiscriminator:')
print(netD)

# x = torch.randn(1, NoiseDimension)
x = torch.ones(1, 3, 64, 64)
fake_y = netG(x).detach()
print(fake_y.shape)
print(netD(fake_y).shape)