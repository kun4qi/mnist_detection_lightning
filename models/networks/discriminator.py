import torch
from torch import nn

class Discriminator(nn.Module):

  def __init__(self, input_dim, dis_filters):
    super(Discriminator, self).__init__()

    def CBA(in_channel, out_channel, kernel_size=4, stride=2, padding=1, activation=nn.LeakyReLU(0.1, inplace=True)):
        seq = []
        seq += [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
        seq += [nn.BatchNorm2d(out_channel)]
        seq += [activation]

        return nn.Sequential(*seq)

    seq = []
    seq += [CBA(input_dim, dis_filters[0])]
    seq += [CBA(dis_filters[0], dis_filters[1])]
    seq += [CBA(dis_filters[1], dis_filters[2])]
    seq += [CBA(dis_filters[2], dis_filters[3])]
    self.feature_network = nn.Sequential(*seq)

    self.critic_network = nn.Conv2d(dis_filters[3], 1, kernel_size=4, stride=1)

  def forward(self, x):
      out = self.feature_network(x)

      feature = out
      feature = feature.view(feature.size(0), -1)

      out = self.critic_network(out)

      return out, feature