import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.blocks import BasicBlock, BottleneckBlock
from src.utils.initialization import initialize_weights

class ResNetPreAct(nn.Module):
    """
    Pre-activation ResNet configurable via un objeto de configuración.

    Atributos esperados en config.model:
      - in_channels (opcional): número de canales de entrada.
      - input_shape (opcional): tupla (batch, canales, H, W).
      - n_classes: número de clases de salida.
      - base_channels: canales base en la primera etapa.
      - block_type: 'basic' o 'bottleneck'.
      - depth: profundidad total del modelo.
      - remove_first_relu: bool para bloques.
      - add_last_bn: bool para bloques.
      - preact_stage: lista de 3 bools para preactivación en cada etapa.
    """
    def __init__(self, config):
        super().__init__()
        # Determinar canales de entrada
        if hasattr(config.model, 'in_channels'):
            in_ch = config.model.in_channels
        else:
            # fallback a input_shape
            in_ch = config.model.input_shape[1]

        n_classes = config.model.n_classes
        base_channels = config.model.base_channels
        block_type = config.model.block_type
        depth = config.model.depth
        remove_first_relu = config.model.remove_first_relu
        add_last_bn = config.model.add_last_bn
        preact_stage = config.model.preact_stage

        # Selección de bloque y número de bloques por etapa
        if block_type == 'basic':
            Block = BasicBlock
            n_blocks = (depth - 2) // 6
        else:
            Block = BottleneckBlock
            n_blocks = (depth - 2) // 9

        # Definición de canales por etapa
        channels = [
            base_channels,
            base_channels * 2 * Block.expansion,
            base_channels * 4 * Block.expansion,
        ]

        # Convolución inicial
        # Siempre inicializamos con el número máximo de canales (RGB)
        self.conv = nn.Conv2d(
            max(1, in_ch), channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        # Etapas residuales
        self.stage1 = self._make_stage(channels[0], channels[0], n_blocks, Block,
                                       stride=1, preact=preact_stage[0],
                                       remove_first_relu=remove_first_relu, add_last_bn=add_last_bn)
        self.stage2 = self._make_stage(channels[0], channels[1], n_blocks, Block,
                                       stride=2, preact=preact_stage[1],
                                       remove_first_relu=remove_first_relu, add_last_bn=add_last_bn)
        self.stage3 = self._make_stage(channels[1], channels[2], n_blocks, Block,
                                       stride=2, preact=preact_stage[2],
                                       remove_first_relu=remove_first_relu, add_last_bn=add_last_bn)

        # BatchNorm final y clasificador
        self.bn = nn.BatchNorm2d(channels[2])
        self.fc = nn.Linear(channels[2], n_classes)

        # Inicialización de pesos
        self.apply(initialize_weights)

    def _make_stage(self, in_ch, out_ch, n_blocks, Block,
                    stride, preact, remove_first_relu, add_last_bn):
        layers = []
        for i in range(n_blocks):
            layers.append(
                Block(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    stride if i == 0 else 1,
                    remove_first_relu=remove_first_relu,
                    add_last_bn=add_last_bn,
                    preact=preact if i == 0 else False
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # Soporte dinámico para imágenes de 1 canal (ej. MNIST)
        if x.shape[1] == 1 and self.conv.weight.shape[1] == 3:
            # duplicar canal para usar pesos RGB
            x = x.repeat(1, 3, 1, 1)
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)
