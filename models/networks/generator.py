from torch import nn


class Generator(nn.Module):

  def __init__(self, z_dim, gen_filters):
    super(Generator, self).__init__()

    def CBA(in_channel, out_channel, kernel_size=4, stride=2, padding=1, activation=nn.ReLU(inplace=True), bn=True):
        seq = []
        seq += [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
        if bn is True:
          seq += [nn.BatchNorm2d(out_channel)]
        seq += [activation]

        return nn.Sequential(*seq)

    seq = []
    seq += [CBA(z_dim, gen_filters[0], stride=1, padding=0)]
    seq += [CBA(gen_filters[0], gen_filters[1])]
    seq += [CBA(gen_filters[1], gen_filters[2])]
    seq += [CBA(gen_filters[2], gen_filters[3])]
    seq += [CBA(gen_filters[3], gen_filters[4], activation=nn.Tanh(), bn=False)]

    self.generator_network = nn.Sequential(*seq)

  def forward(self, z):
      out = self.generator_network(z)

      return out