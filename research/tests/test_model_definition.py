import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from model_definition import ECGClassifier


class ModelCheck(unittest.TestCase):
    def test_finite_logits_and_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        model=ECGClassifier()
        logits=model(torch.randn(3,1,1000))
        self.assertEqual(tuple(logits.shape),(3,))
        self.assertTrue(torch.isfinite(logits).all())
        torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.,0.])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__':unittest.main()
