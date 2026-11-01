from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from policies.policy import Policy, weight_init
from policies.nmlinear import NMLinearReduced


class NMEpiCaviaCategoricalMLPPolicy(Policy):
    def __init__(
        self,
        input_size,
        output_size,
        device,
        hidden_sizes=(),
        num_context_params=10,
        nonlinearity=F.relu,
        nm_size=5,
        nm_gate="hard",
        epi_size=8,
        epi_decay=0.2,
        epi_strength=1.0,
    ):
        super(NMEpiCaviaCategoricalMLPPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size
        )

        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity

        self.nm_size = nm_size
        assert nm_gate in ["hard", "soft"]
        self.nm_gate = nm_gate

        self.num_layers = len(hidden_sizes) + 1
        self.num_context_params = num_context_params

        self.epi_size = epi_size
        self.epi_decay = epi_decay
        self.epi_strength = epi_strength

        layer_sizes = (
            input_size,
        ) + hidden_sizes + (
            output_size,
        )

        self.add_module(
            "layer1",
            NMLinearReduced(
                layer_sizes[0] + num_context_params,
                layer_sizes[1],
                nm_size,
                nm_gate
            )
        )

        for i in range(2, self.num_layers):
            self.add_module(
                "layer{0}".format(i),
                NMLinearReduced(
                    layer_sizes[i - 1],
                    layer_sizes[i],
                    nm_size,
                    nm_gate
                )
            )

        self.add_module(
            "layer{0}".format(self.num_layers),
            nn.Linear(
                layer_sizes[self.num_layers - 1],
                layer_sizes[self.num_layers]
            )
        )

        self.epi_encoder = nn.Linear(
            num_context_params,
            epi_size
        )

        for i in range(1, self.num_layers):
            self.add_module(
                "epi_gate{0}".format(i),
                nn.Linear(
                    epi_size,
                    layer_sizes[i]
                )
            )

        self.context_params = torch.zeros(
            self.num_context_params,
            requires_grad=True,
            device=self.device
        )

        self.epi_state = torch.zeros(
            self.epi_size,
            device=self.device
        )

        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        context = self.context_params.expand(
            input.shape[:-1] + self.context_params.shape
        )

        output = torch.cat(
            (input, context),
            dim=len(input.shape) - 1
        )

        for i in range(1, self.num_layers):
            output_std = F.linear(
                output,
                weight=params[
                    "layer{0}.std.weight".format(i)
                ],
                bias=params[
                    "layer{0}.std.bias".format(i)
                ]
            )

            mod_features = self.nonlinearity(
                F.linear(
                    output,
                    weight=params[
                        "layer{0}.in_nm.weight".format(i)
                    ],
                    bias=params[
                        "layer{0}.in_nm.bias".format(i)
                    ]
                )
            )

            nm_gate = torch.tanh(
                F.linear(
                    mod_features,
                    weight=params[
                        "layer{0}.out_nm.weight".format(i)
                    ],
                    bias=params[
                        "layer{0}.out_nm.bias".format(i)
                    ]
                )
            )

            if self.nm_gate == "hard":
                nm_gate = torch.sign(nm_gate)
                nm_gate[nm_gate == 0.0] = 1.0

            nm_output = self.nonlinearity(
                output_std * nm_gate
            )

            epi_gate = torch.tanh(
                F.linear(
                    self.epi_state,
                    weight=params[
                        "epi_gate{0}.weight".format(i)
                    ],
                    bias=params[
                        "epi_gate{0}.bias".format(i)
                    ]
                )
            )

            epi_modulation = (
                1.0
                + self.epi_strength * epi_gate
            )

            output = (
                nm_output
                * epi_modulation
            )

        logits = F.linear(
            output,
            weight=params[
                "layer{0}.weight".format(
                    self.num_layers
                )
            ],
            bias=params[
                "layer{0}.bias".format(
                    self.num_layers
                )
            ]
        )

        return Categorical(logits=logits)

    def update_params(
        self,
        loss,
        step_size,
        first_order=False,
        params=None
    ):
        if params is None:
            params = OrderedDict(
                self.named_parameters()
            )

        grads = torch.autograd.grad(
            loss,
            self.context_params,
            create_graph=not first_order
        )[0]

        if not first_order:
            self.context_params = (
                self.context_params
                - step_size * grads
            )
        else:
            self.context_params = (
                self.context_params
                - step_size * grads.detach()
            )

        epi_candidate = torch.tanh(
            F.linear(
                self.context_params,
                weight=params[
                    "epi_encoder.weight"
                ],
                bias=params[
                    "epi_encoder.bias"
                ]
            )
        )

        self.epi_state = (
            (1.0 - self.epi_decay)
            * self.epi_state
            + self.epi_decay
            * epi_candidate
        )

        return OrderedDict(
            self.named_parameters()
        )

    def reset_context(self):
        self.context_params = torch.zeros(
            self.num_context_params,
            requires_grad=True,
            device=self.device
        )

        self.epi_state = torch.zeros(
            self.epi_size,
            device=self.device
        )

    def forward_analysismode(
        self,
        input,
        params=None
    ):
        if params is None:
            params = OrderedDict(
                self.named_parameters()
            )

        context = self.context_params.expand(
            input.shape[:-1]
            + self.context_params.shape
        )

        output = torch.cat(
            (input, context),
            dim=len(input.shape) - 1
        )

        activations = [
            context,
            self.epi_state.expand(
                input.shape[:-1]
                + self.epi_state.shape
            )
        ]

        for i in range(1, self.num_layers):
            output_std = F.linear(
                output,
                weight=params[
                    "layer{0}.std.weight".format(i)
                ],
                bias=params[
                    "layer{0}.std.bias".format(i)
                ]
            )

            mod_features = self.nonlinearity(
                F.linear(
                    output,
                    weight=params[
                        "layer{0}.in_nm.weight".format(i)
                    ],
                    bias=params[
                        "layer{0}.in_nm.bias".format(i)
                    ]
                )
            )

            nm_gate = torch.tanh(
                F.linear(
                    mod_features,
                    weight=params[
                        "layer{0}.out_nm.weight".format(i)
                    ],
                    bias=params[
                        "layer{0}.out_nm.bias".format(i)
                    ]
                )
            )

            if self.nm_gate == "hard":
                nm_gate = torch.sign(nm_gate)
                nm_gate[nm_gate == 0.0] = 1.0

            nm_gate_copy = copy.deepcopy(
                nm_gate
            )

            nm_output = self.nonlinearity(
                output_std * nm_gate
            )

            epi_gate = torch.tanh(
                F.linear(
                    self.epi_state,
                    weight=params[
                        "epi_gate{0}.weight".format(i)
                    ],
                    bias=params[
                        "epi_gate{0}.bias".format(i)
                    ]
                )
            )

            epi_modulation = (
                1.0
                + self.epi_strength
                * epi_gate
            )

            output = (
                nm_output
                * epi_modulation
            )

            activations.append(
                output_std
            )

            activations.append(
                mod_features
            )

            activations.append(
                nm_gate_copy
            )

            activations.append(
                nm_output
            )

            activations.append(
                epi_gate.expand(
                    input.shape[:-1]
                    + epi_gate.shape
                )
            )

            activations.append(
                output
            )

        logits = F.linear(
            output,
            weight=params[
                "layer{0}.weight".format(
                    self.num_layers
                )
            ],
            bias=params[
                "layer{0}.bias".format(
                    self.num_layers
                )
            ]
        )

        activations.append(logits)

        return (
            Categorical(logits=logits),
            torch.cat(
                activations,
                dim=-1
            )
        )
