# Evidential Item Response Theory

## Abstract
In cognitive diagnosis, understanding student ability with uncertainty is critical for personalized education and reliable decision-making. In this work, we propose Evidential Item Response Theory (EvidentIRT) to capture the aleatoric and epistemic uncertainty of student ability. EvidentIRT is a novel variational autoencoder for item response data that models student abilities as latent variables via the Normal-Inverse Gamma (NIG) distribution. Specifically, given a student’s observed response vector, our encoder predicts item-wise local evidential estimates of ability, which are summed via an approximate strategy into a global posterior over the latent ability. This allows the model to adopt an inductive learning fashion where the student's ability is predicted by a deterministic neural network rather than being estimated as the parameters of the embedding layer. The decoder then reconstructs the response pattern using this uncertain latent representation, enabling the inference under data sparsity and noise.  The loss function is designed to penalize confident predictions on incorrect reconstructions and regularize with a zero-evidence prior, encouraging calibrated uncertainty estimation. Experiments on real-world educational datasets demonstrate that our approach not only yields accurate response predictions but also provides meaningful uncertainty quantification aligned with response coverage and other factors.



### Run EvidentIRT
To fit a EvidentIRT model, use the following command:
```bash
nohup python -u src/torch_core/evidirt.py --irt-model 2pl --dataset math1 --lr 0.0005 --batch-size 128 --hidden-dim 32 --gpu-device 0 --cuda > evidirt_math1_2pl_2.log 2>&1 &
```

```bash
nohup python -u src/torch_core/evidirt.py --irt-model dina --dataset math1 --lr 0.0005 --batch-size 128 --hidden-dim 32 --gpu-device 0 --cuda > evidirt_math1_2pl_2.log 2>&1 &
```

```bash
nohup python -u src/torch_core/evidirt.py --irt-model neural --dataset math2 --lr 0.0005 --batch-size 128 --hidden-dim 32 --gpu-device 0 --cuda > evidirt_math1_2pl_2.log 2>&1 &
```

If you don't want to use Q-matrix, set ability-dim = 1; else set to the number of concepts.

### Parameters

```python
    parser.add_argument('--irt-model', type=str, default='neural',
                        choices=['1pl', '2pl', '3pl', 'dina', 'neural'],
                        help='1pl|2pl|3pl (default: 1pl)')
    parser.add_argument('--dataset', type=str, default='math1',
                        help='which dataset to run on (default: math1)')
    parser.add_argument('--ability-dim', type=int, default=11,
                        help='number of ability dimensions (default: 1)')
    parser.add_argument('--ability-merge', type=str, default='sum',
                        choices=['sum'],
                        help='mean|product|transformer (default: product)')
    parser.add_argument('--generative-model', type=str, default='irt', 
                        choices=['irt', 'link', 'deep', 'residual'],
                        help='irt|link|deep|residual|neural (default: irt)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='number of hidden dims (default: 64)')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='default learning rate: 5e-3')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--gpu-device', type=int, default=0, 
                        help='which CUDA device to use (default: 0)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='seed (default: 42)')
```

