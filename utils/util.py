import torch
import json
import collections

def load_json(path):
    def _json_object_hook(d):
        return collections.namedtuple('X', d.keys())(*d.values())
    def _json_to_obj(data):
        return json.loads(data, object_hook=_json_object_hook)
    return _json_to_obj(open(path).read())

def anomaly_score(input_image, fake_image, D):
  # Residual loss の計算
  residual_loss = torch.sum(torch.abs(input_image - fake_image), (1, 2, 3))

  # Discrimination loss の計算
  _, real_feature = D(input_image)
  _, fake_feature = D(fake_image)
  discrimination_loss = torch.sum(torch.abs(real_feature - fake_feature), (1))

  # 二つのlossを一定の割合で足し合わせる
  total_loss_by_image = 0.9 * residual_loss + 0.1 * discrimination_loss
  total_loss = total_loss_by_image.sum()

  return total_loss, total_loss_by_image, residual_loss