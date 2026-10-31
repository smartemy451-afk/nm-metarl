from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from policies.policy import Policy, weight_init


class EpiCaviaCategoricalMLPPolicy(Policy):
    """
    CAVIA policy with a persistent epigenetic-inspired regulatory state.

    context_params:
        fast CAVIA adaptation variables.

    epi_state:
        slower persistent regulatory state, updated only after
        an inner-loop adaptation step.

    Before adaptation epi_state == 0, therefore modulation == 1
    and the network behaves like standard CAVIA.
    """

    def __init__(
        self,
        input_size,
        output_size,
        device,
        hidden_sizes=(),
        num_context_params=10,
        nonlinearity=F.relu,
        epi_size=8,
        epi_decay=0.2,
        epi_strength=1.0,
    ):
        super(EpiCaviaCategoricalMLPPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size
        )

        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity

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

        # ---------------------------------------------------------
        # Normal CAVIA network
        # First layer receives observation + context
        # ---------------------------------------------------------

        self.add_module(
            "layer1",
            nn.Linear(
                layer_sizes[0] + num_context_params,
                layer_sizes[1]
            )
        )

        for i in range(2, self.num_layers + 1):
            self.add_module(
                "layer{0}".format(i),
                nn.Linear(
                    layer_sizes[i - 1],
                    layer_sizes[i]
                )
            )

        # ---------------------------------------------------------
        # Epigenetic regulator
        #
        # context -> epigenetic state
        # ---------------------------------------------------------

        self.epi_encoder = nn.Linear(
            num_context_params,
            epi_size
        )

        # Each hidden layer receives a regulatory gate
        for i in range(1, self.num_layers):
            self.add_module(
                "epi_gate{0}".format(i),
                nn.Linear(
                    epi_size,
                    layer_sizes[i]
                )
            )

        # Fast CAVIA context
        self.context_params = torch.zeros(
            self.num_context_params,
            requires_grad=True,
            device=self.device
        )

        # Persistent regulatory state
        self.epi_state = torch.zeros(
            self.epi_size,
            device=self.device
        )

        self.apply(weight_init)

    # =============================================================
    # FORWARD
    # =============================================================

    def forward(self, input, params=None):

        if params is None:
            params = OrderedDict(self.named_parameters())

        # Concatenate CAVIA context with observation
        context = self.context_params.expand(
            input.shape[:-1] + self.context_params.shape
        )

        output = torch.cat(
            (input, context),
            dim=len(input.shape) - 1
        )

        # Hidden layers
        for i in range(1, self.num_layers):

            # Standard neural activity
            output_std = F.linear(
                output,
                weight=params[
                    "layer{0}.weight".format(i)
                ],
                bias=params[
                    "layer{0}.bias".format(i)
                ]
            )

            # Epigenetic regulatory signal
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

            # Residual modulation:
            #
            # epi_state = 0
            #       =>
            # gate = 0
            #       =>
            # modulation = 1
            #
            modulation = (
                1.0
                + self.epi_strength * epi_gate
            )

            output = self.nonlinearity(
                output_std * modulation
            )

        # Output layer is NOT epigenetically gated
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

    # =============================================================
    # INNER-LOOP ADAPTATION
    # =============================================================

    def update_params(
        self,
        loss,
        step_size,
        first_order=False,
        params=None
    ):
        """
        1. Update CAVIA context.
        2. Convert adapted context into epigenetic signal.
        3. Update persistent epigenetic state.
        """

        if params is None:
            params = OrderedDict(
                self.named_parameters()
            )

        # Gradient only with respect to CAVIA context
        grads = torch.autograd.grad(
            loss,
            self.context_params,
            create_graph=not first_order
        )[0]

        # Standard CAVIA context update
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

        # ---------------------------------------------------------
        # EPIGENETIC UPDATE
        #
        # adapted context -> candidate epi state
        # ---------------------------------------------------------

        epi_candidate = torch.tanh(
            F.linear(
                self.context_params,
                weight=params["epi_encoder.weight"],
                bias=params["epi_encoder.bias"]
            )
        )

        # Persistent / history-dependent update
        self.epi_state = (
            (1.0 - self.epi_decay)
            * self.epi_state
            + self.epi_decay
            * epi_candidate
        )

        # CAVIA does not update network parameters in inner loop
        return OrderedDict(
            self.named_parameters()
        )

    # =============================================================
    # RESET BETWEEN TASKS
    # =============================================================

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

    # =============================================================
    # ANALYSIS MODE
    # =============================================================

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

        activations = [context]

        # Include epigenetic state for later analysis
        epi_expanded = self.epi_state.expand(
            input.shape[:-1]
            + self.epi_state.shape
        )

        activations.append(epi_expanded)

        for i in range(1, self.num_layers):

            output_std = F.linear(
                output,
                weight=params[
                    "layer{0}.weight".format(i)
                ],
                bias=params[
                    "layer{0}.bias".format(i)
                ]
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

            modulation = (
                1.0
                + self.epi_strength
                * epi_gate
            )

            output = self.nonlinearity(
                output_std * modulation
            )

            gate_expanded = epi_gate.expand(
                input.shape[:-1]
                + epi_gate.shape
            )

            activations.append(output_std)
            activations.append(gate_expanded)
            activations.append(output)

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
            torch.cat(activations, dim=-1)
        )